import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .layers import DrugGraphEncoder, GARTransformer


NUM_TOKENS = 10
DRUG_A_SLICE = slice(1, 4)
DRUG_B_SLICE = slice(4, 7)
CELL_SLICE = slice(7, 10)


def token_entity(idx):
    if idx == 0:
        return "syn"
    if 1 <= idx <= 3:
        return "druga"
    if 4 <= idx <= 6:
        return "drugb"
    return "cell"


def build_interaction_type_matrix(device):
    mat = torch.zeros(NUM_TOKENS, NUM_TOKENS, dtype=torch.long)
    for i in range(NUM_TOKENS):
        for j in range(NUM_TOKENS):
            if i == 0 or j == 0:
                mat[i, j] = 0
            else:
                ei, ej = token_entity(i), token_entity(j)
                if ei == ej:
                    mat[i, j] = 4 if ei == "cell" else 1
                elif {ei, ej} == {"druga", "drugb"}:
                    mat[i, j] = 2
                elif (ei in ("druga", "drugb") and ej == "cell") or (ei == "cell" and ej in ("druga", "drugb")):
                    mat[i, j] = 3
    return mat.to(device)


class GARSynNet(nn.Module):
    def __init__(self, model_config, features):
        super().__init__()
        hidden = model_config["hidden_size"]
        dropout = model_config["dropout"]
        self.hidden_size = hidden
        self.use_shared_drug_embedding = model_config.get("use_shared_drug_embedding", False)
        self.graph_cache_mode = model_config.get("graph_cache_mode", "batch")
        self._type_matrix = None
        self.idx_to_graph = None
        self.all_graph_batch = None
        self.graph_index_map = None

        self.register_buffer("drug_fp", torch.tensor(np.array(features.drug_fp), dtype=torch.float32))
        self.register_buffer("drug_seq", torch.tensor(np.array(features.drug_seq), dtype=torch.float32))
        self.register_buffer("cell_exp", torch.tensor(np.array(features.cell_exp), dtype=torch.float32))
        self.register_buffer("cell_mu", torch.tensor(np.array(features.cell_mu), dtype=torch.float32))
        self.register_buffer("cell_nv", torch.tensor(np.array(features.cell_nv), dtype=torch.float32))
        self.drug_graph = features.drug_graph

        self.graph_encoder = DrugGraphEncoder(model_config["layer_drug"], model_config["dim_drug_gnn"])
        self.drug_fc_fp = nn.Linear(model_config["dim_drug_fp"], hidden)
        self.drug_fc_seq = nn.Linear(model_config["dim_drug_seq"], hidden)
        self.drug_fc_gra = nn.Linear(model_config["dim_drug_gra"], hidden)
        self.cell_fc_exp = nn.Linear(model_config["dim_cell_exp"], hidden)
        self.cell_fc_mu = nn.Linear(model_config["dim_cell_mu"], hidden)
        self.cell_fc_nv = nn.Linear(model_config["dim_cell_nv"], hidden)

        for module in [
            self.drug_fc_fp,
            self.drug_fc_seq,
            self.drug_fc_gra,
            self.cell_fc_exp,
            self.cell_fc_mu,
            self.cell_fc_nv,
        ]:
            nn.init.kaiming_normal_(module.weight)

        self.syn_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.syn_token, std=0.02)

        self.modality_embedding = nn.Embedding(7, hidden)
        self.entity_embedding = nn.Embedding(4, hidden)
        self.role_embedding = nn.Embedding(3, hidden)
        self.position_embedding = nn.Embedding(NUM_TOKENS, hidden)
        self.input_norm = nn.LayerNorm(hidden)

        modality_ids = [0, 1, 2, 3, 1, 2, 3, 4, 5, 6]
        role_ids = [0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        if self.use_shared_drug_embedding:
            entity_ids = [0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
            pos_ids = [0, 1, 2, 3, 1, 2, 3, 4, 5, 6]
        else:
            entity_ids = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
            pos_ids = list(range(NUM_TOKENS))

        self.register_buffer("modality_ids", torch.tensor(modality_ids, dtype=torch.long))
        self.register_buffer("entity_ids", torch.tensor(entity_ids, dtype=torch.long))
        self.register_buffer("role_ids", torch.tensor(role_ids, dtype=torch.long))
        self.register_buffer("pos_ids", torch.tensor(pos_ids, dtype=torch.long))

        self.gar_encoder = GARTransformer(
            hidden_size=hidden,
            num_heads=model_config["num_attention_heads"],
            num_layers=model_config["num_gar_layers"],
            dropout=dropout,
            ffn_mult=model_config["gar_ffn_mult"],
            use_interaction_type_bias=model_config["use_interaction_type_bias"],
            use_gate=model_config.get("use_gate", True),
            use_depth_residual=model_config.get("use_depth_residual", True),
            num_interaction_types=model_config["num_interaction_types"],
        )

        self.synergy_predictor = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        for module in self.synergy_predictor:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

    @property
    def device(self):
        return self.syn_token.device

    def build_type_matrix(self):
        if self._type_matrix is None or self._type_matrix.device != self.device:
            self._type_matrix = build_interaction_type_matrix(self.device)
        return self._type_matrix

    def build_idx_to_graph(self, reverse_drug_map):
        self.idx_to_graph = {}
        for idx, name in reverse_drug_map.items():
            self.idx_to_graph[int(idx)] = self.drug_graph[name].to(self.device)

    def build_all_graph_batch(self, reverse_drug_map):
        ordered = sorted((int(idx), name) for idx, name in reverse_drug_map.items())
        graphs = [self.drug_graph[name].to(self.device) for _, name in ordered]
        self.all_graph_batch = Batch.from_data_list(graphs).to(self.device)

        max_idx = max(idx for idx, _ in ordered)
        index_map = torch.empty(max_idx + 1, dtype=torch.long, device=self.device)
        for row, (idx, _) in enumerate(ordered):
            index_map[idx] = row
        self.graph_index_map = index_map

    def graph_batch(self, reverse_drug_map, drug_indices):
        if self.idx_to_graph is None:
            self.build_idx_to_graph(reverse_drug_map)
        any_graph = next(iter(self.idx_to_graph.values()))
        if any_graph.x.device != self.device:
            self.build_idx_to_graph(reverse_drug_map)
        return [self.idx_to_graph[int(idx)] for idx in drug_indices.detach().cpu().tolist()]

    def encode_drug(self, reverse_drug_map, drug_indices):
        drug_indices = drug_indices.long()
        batch_size = drug_indices.shape[0]
        drug_fp = self.drug_fp[drug_indices].view(batch_size, 1, -1)
        drug_seq = self.drug_seq[drug_indices].view(batch_size, 1, -1)
        if self.graph_cache_mode == "full_batch":
            if self.all_graph_batch is None or self.all_graph_batch.x.device != self.device:
                self.build_all_graph_batch(reverse_drug_map)
            all_gra = self.graph_encoder(self.all_graph_batch)
            drug_gra = all_gra.index_select(0, self.graph_index_map[drug_indices])
        else:
            drug_gra = self.graph_encoder(self.graph_batch(reverse_drug_map, drug_indices))
        return (
            self.drug_fc_fp(drug_fp).squeeze(1),
            self.drug_fc_seq(drug_seq).squeeze(1),
            self.drug_fc_gra(drug_gra),
        )

    def encode_two_drugs(self, reverse_drug_map, drug_a_idx, drug_b_idx):
        all_idx = torch.cat([drug_a_idx.long(), drug_b_idx.long()], dim=0)
        unique_idx, inverse = torch.unique(all_idx, sorted=False, return_inverse=True)
        fp_u, seq_u, gra_u = self.encode_drug(reverse_drug_map, unique_idx)
        batch_size = drug_a_idx.shape[0]
        inv_a = inverse[:batch_size]
        inv_b = inverse[batch_size:]
        return fp_u[inv_a], seq_u[inv_a], gra_u[inv_a], fp_u[inv_b], seq_u[inv_b], gra_u[inv_b]

    def encode_cell(self, cell_indices):
        cell_indices = cell_indices.long()
        batch_size = cell_indices.shape[0]
        cell_exp = self.cell_exp[cell_indices].view(batch_size, 1, -1)
        cell_mu = self.cell_mu[cell_indices].view(batch_size, 1, -1)
        cell_nv = self.cell_nv[cell_indices].view(batch_size, 1, -1)
        return (
            self.cell_fc_exp(cell_exp).squeeze(1),
            self.cell_fc_mu(cell_mu).squeeze(1),
            self.cell_fc_nv(cell_nv).squeeze(1),
        )

    def construct_tokens(self, da_fp, da_seq, da_gra, db_fp, db_seq, db_gra, c_exp, c_mut, c_cnv):
        batch_size = da_fp.shape[0]
        syn = self.syn_token.expand(batch_size, 1, self.hidden_size)
        tokens = torch.stack(
            [
                syn.squeeze(1),
                da_fp,
                da_seq,
                da_gra,
                db_fp,
                db_seq,
                db_gra,
                c_exp,
                c_mut,
                c_cnv,
            ],
            dim=1,
        )
        x = (
            tokens
            + self.modality_embedding(self.modality_ids)
            + self.entity_embedding(self.entity_ids)
            + self.role_embedding(self.role_ids)
            + self.position_embedding(self.pos_ids)
        )
        return self.input_norm(x)

    def forward_from_embeddings(
        self,
        da_fp,
        da_seq,
        da_gra,
        db_fp,
        db_seq,
        db_gra,
        c_exp,
        c_mut,
        c_cnv,
        return_aux=False,
    ):
        x = self.construct_tokens(da_fp, da_seq, da_gra, db_fp, db_seq, db_gra, c_exp, c_mut, c_cnv)
        type_matrix = self.build_type_matrix()
        if return_aux:
            h, aux = self.gar_encoder(x, type_matrix=type_matrix, return_aux=True)
            aux["type_matrix"] = type_matrix
        else:
            h = self.gar_encoder(x, type_matrix=type_matrix, return_aux=False)
            aux = None

        h_syn = h[:, 0, :]
        h_a = h[:, DRUG_A_SLICE, :].mean(dim=1)
        h_b = h[:, DRUG_B_SLICE, :].mean(dim=1)
        h_c = h[:, CELL_SLICE, :].mean(dim=1)
        h_pair = torch.cat([h_a + h_b, torch.abs(h_a - h_b), h_a * h_b, h_c, h_syn], dim=-1)
        pred = self.synergy_predictor(h_pair)
        if return_aux:
            return pred, aux
        return pred

    def forward(self, reverse_drug_map, triplets, return_aux=False):
        triplets = triplets.long().to(self.device)
        da = triplets[:, 0]
        db = triplets[:, 1]
        cell = triplets[:, 2]
        da_fp, da_seq, da_gra, db_fp, db_seq, db_gra = self.encode_two_drugs(reverse_drug_map, da, db)
        c_exp, c_mut, c_cnv = self.encode_cell(cell)
        return self.forward_from_embeddings(
            da_fp, da_seq, da_gra, db_fp, db_seq, db_gra, c_exp, c_mut, c_cnv, return_aux=return_aux
        )

    def forward_with_perm(self, reverse_drug_map, triplets, return_aux=True):
        triplets = triplets.long().to(self.device)
        da = triplets[:, 0]
        db = triplets[:, 1]
        cell = triplets[:, 2]
        da_fp, da_seq, da_gra, db_fp, db_seq, db_gra = self.encode_two_drugs(reverse_drug_map, da, db)
        c_exp, c_mut, c_cnv = self.encode_cell(cell)
        if return_aux:
            pred, aux = self.forward_from_embeddings(
                da_fp, da_seq, da_gra, db_fp, db_seq, db_gra, c_exp, c_mut, c_cnv, return_aux=True
            )
        else:
            pred = self.forward_from_embeddings(
                da_fp, da_seq, da_gra, db_fp, db_seq, db_gra, c_exp, c_mut, c_cnv, return_aux=False
            )
            aux = None
        pred_swap = self.forward_from_embeddings(
            db_fp, db_seq, db_gra, da_fp, da_seq, da_gra, c_exp, c_mut, c_cnv, return_aux=False
        )
        return pred, pred_swap, aux


def extract_interpretability(model, reverse_drug_map, triplets):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred, aux = model(reverse_drug_map, triplets, return_aux=True)
        last_gate = aux["gate"][-1]
        modality_gate_score = last_gate.mean(dim=(2, 3))
        type_matrix = aux["type_matrix"]
        last_attn = aux["attn"][-1]
        num_types = int(type_matrix.max().item()) + 1
        batch_size, heads, num_tokens, _ = last_attn.shape
        type_attn = torch.zeros(batch_size, heads, num_types, device=last_attn.device)
        for rel_type in range(num_types):
            mask = (type_matrix == rel_type).float()
            denom = mask.sum() + 1e-8
            type_attn[:, :, rel_type] = (last_attn * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3)) / denom
        result = {
            "prediction": pred.detach().cpu(),
            "modality_gate_score": modality_gate_score.detach().cpu(),
            "interaction_type_attention": type_attn.mean(dim=1).detach().cpu(),
            "depth_residual_weights": [a.detach().cpu() for a in aux["depth_alpha"]],
        }
    if was_training:
        model.train()
    return result
