import math, random
import copy
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from util.misc import inverse_sigmoid
from .utils import gen_encoder_output_proposals, MLP, _get_activation_fn, gen_sineembed_for_position
from .ops.modules import MSDeformAttn
from .dn_components import prepare_for_cdn, dn_post_process

from .query_allocator import AdaptiveQueryBudgetAllocator
from .feature_calibrator import DensityGuidedFeatureCalibrator, DensityPyramidAdapter
from .proposal_selection import select_proposal_indices


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8,
                 num_queries=300,
                 num_encoder_layers=6,
                 num_unicoder_layers=0,
                 num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 # for deformable encoder
                 deformable_encoder=False,
                 deformable_decoder=False,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 # init query
                 learnable_tgt_init=False,
                 decoder_query_perturber=None,
                 add_channel_attention=False,
                 add_pos_value=False,
                 random_refpoints_xy=False,
                 # two stage
                 two_stage_type='standard',
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 # evo of #anchors
                 dec_layer_number=None,
                 rm_enc_query_scale=True,
                 rm_dec_query_scale=True,
                 rm_self_attn_layers=None,
                 key_aware_type=None,
                 # layer share
                 layer_share_type=None,
                 # for detach
                 rm_detach=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 # for dn
                 embed_init_tgt=False,
                 use_detached_boxes_dec_out=False,

                 query_budget_levels=None,
                 allocator_fallback_queries=900,
                 use_ema=True,
                 ema_decay=0.9997,
                 allocator_use_boundary_ema=True,
                 calibrator_gate_type='tanh',
                 calibrator_use_spatial=True,
                 calibrator_spatial_alphas=None,
                 proposal_selection_mode='fused',
                 proposal_density_weight=0.25,
                 mixed_density_ratio=0.25,
                 grouped_decoder_inference=True,
                 force_query_budget=None,
                 ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.query_budget_levels = list(query_budget_levels or [300, 500, 900, 1500])
        self.num_budget_levels = len(self.query_budget_levels)
        self.proposal_selection_mode = proposal_selection_mode
        self.proposal_density_weight = float(proposal_density_weight)
        self.mixed_density_ratio = float(mixed_density_ratio)
        self.grouped_decoder_inference = bool(grouped_decoder_inference)
        self.force_query_budget = force_query_budget
        self.current_epoch = 0
        assert query_dim == 4

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"
        if use_deformable_box_attn:
            assert deformable_encoder or deformable_encoder

        assert layer_share_type in [None, 'encoder', 'decoder', 'both']
        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # choose encoder layer type
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, enc_n_points,
                                                              add_channel_attention=add_channel_attention,
                                                              use_deformable_box_attn=use_deformable_box_attn,
                                                              box_attn_type=box_attn_type)
        else:
            raise NotImplementedError
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.query_allocator = AdaptiveQueryBudgetAllocator(
            feature_dim=256,
            query_budget_levels=self.query_budget_levels,
            max_objects=1500,
            fallback_queries=allocator_fallback_queries,
            use_ema=allocator_use_boundary_ema,
            ema_decay=ema_decay,
        )

        self.feature_calibrator = DensityGuidedFeatureCalibrator(
            gate_channels=256,
            reduction_ratio=16,
            num_feature_levels=self.num_feature_levels,
            gate_type=calibrator_gate_type,
            use_spatial=calibrator_use_spatial,
            level_spatial_alphas=calibrator_spatial_alphas,
        )
        self.density_pyramid = DensityPyramidAdapter(is_5_scale=True)

        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers,
            encoder_norm, d_model=d_model,
            num_queries=num_queries,
            deformable_encoder=deformable_encoder,
            enc_layer_share=enc_layer_share,
            two_stage_type=two_stage_type
        )

        # choose decoder layer type
        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, dec_n_points,
                                                              use_deformable_box_attn=use_deformable_box_attn,
                                                              box_attn_type=box_attn_type,
                                                              key_aware_type=key_aware_type,
                                                              decoder_sa_type=decoder_sa_type,
                                                              module_seq=module_seq)

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim,
                                          modulate_hw_attn=modulate_hw_attn,
                                          num_feature_levels=num_feature_levels,
                                          deformable_decoder=deformable_decoder,
                                          decoder_query_perturber=decoder_query_perturber,
                                          dec_layer_number=dec_layer_number, rm_dec_query_scale=rm_dec_query_scale,
                                          dec_layer_share=dec_layer_share,
                                          use_detached_boxes_dec_out=use_detached_boxes_dec_out
                                          )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type == 'standard':
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(two_stage_pat_embed, d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type == 'no':
            self.init_ref_points(num_queries)

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[0] == num_queries
            else:
                assert dec_layer_number[0] == num_queries * num_patterns

        self._reset_parameters()
        # Parent Xavier initialization must not override AQBA's low-variance heads.
        # This runs before any checkpoint loading, never after warm-start/resume.
        self.query_allocator._init_weights()

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            print("Removing the self-attn in {} decoder layers".format(rm_self_attn_layers))
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    @staticmethod
    def flatten_density_prior(density_prior, spatial_shapes, padding_mask=None):
        levels = []
        for height, width in spatial_shapes.tolist():
            resized = F.interpolate(
                density_prior, size=(int(height), int(width)),
                mode='bilinear', align_corners=False)
            levels.append(resized.flatten(2).squeeze(1))
        flattened = torch.cat(levels, dim=1)
        if padding_mask is not None:
            flattened = flattened.masked_fill(padding_mask, 0.0)
        return flattened

    def select_proposal_indices(self, class_logits, density_prior, padding_mask, topk, proposal_boxes=None):
        return select_proposal_indices(
            class_logits, density_prior, padding_mask, topk,
            mode=self.proposal_selection_mode,
            density_weight=self.proposal_density_weight,
            mixed_density_ratio=self.mixed_density_ratio, proposal_boxes=proposal_boxes)

    @staticmethod
    def _build_target_padding_mask(tgt, query_valid_mask):
        prefix_length = tgt.shape[1] - query_valid_mask.shape[1]
        if prefix_length < 0:
            raise ValueError("query_valid_mask is longer than decoder target sequence")
        if prefix_length == 0:
            return ~query_valid_mask
        prefix = torch.zeros(
            query_valid_mask.shape[0], prefix_length,
            dtype=torch.bool, device=query_valid_mask.device)
        return torch.cat([prefix, ~query_valid_mask], dim=1)

    def _decode_grouped_inference(
            self, output_memory, output_proposals, class_logits, coord_logits,
            density_prior, padding_mask, memory, position, level_start_index,
            spatial_shapes, valid_ratios, query_counts):
        """Run one decoder call per occupied query-budget level.

        Backbone, encoder, allocator and feature calibration remain batched. Only
        decoder tensors are sliced, so a 300-query image never executes padded
        900/1500-query decoder positions.
        """
        batch_size = output_memory.shape[0]
        max_queries = int(query_counts.max().item())
        hidden_size = output_memory.shape[-1]
        grouped_results = []

        for query_count in torch.unique(query_counts, sorted=True).tolist():
            sample_indices = torch.nonzero(
                query_counts == int(query_count), as_tuple=False).flatten()
            if sample_indices.numel() == 0:
                continue
            count = int(query_count)
            group_class = class_logits.index_select(0, sample_indices)
            group_density = density_prior.index_select(0, sample_indices)
            group_padding = padding_mask.index_select(0, sample_indices)
            proposal_indices = self.select_proposal_indices(
                group_class, group_density, group_padding, count,
                proposal_boxes=output_proposals.index_select(0, sample_indices))

            group_coords = coord_logits.index_select(0, sample_indices)
            group_output_memory = output_memory.index_select(0, sample_indices)
            group_output_proposals = output_proposals.index_select(0, sample_indices)
            references_undetached = torch.gather(
                group_coords, 1, proposal_indices.unsqueeze(-1).expand(-1, -1, 4))
            targets_undetached = torch.gather(
                group_output_memory, 1,
                proposal_indices.unsqueeze(-1).expand(-1, -1, hidden_size))
            initial_boxes = torch.gather(
                group_output_proposals, 1,
                proposal_indices.unsqueeze(-1).expand(-1, -1, 4)).sigmoid()

            if self.embed_init_tgt:
                targets = self.tgt_embed.weight[:count].unsqueeze(0).expand(
                    sample_indices.numel(), -1, -1)
            else:
                targets = targets_undetached.detach()

            hidden_states, references = self.decoder(
                tgt=targets.transpose(0, 1),
                memory=memory.index_select(0, sample_indices).transpose(0, 1),
                memory_key_padding_mask=padding_mask.index_select(0, sample_indices),
                pos=position.index_select(0, sample_indices).transpose(0, 1),
                refpoints_unsigmoid=references_undetached.detach().transpose(0, 1),
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios.index_select(0, sample_indices),
            )
            grouped_results.append((
                sample_indices, hidden_states, references,
                targets_undetached, references_undetached, initial_boxes))

        decoder_layers = len(grouped_results[0][1])
        reference_layers = len(grouped_results[0][2])
        hidden_states = [output_memory.new_zeros(batch_size, max_queries, hidden_size)
                         for _ in range(decoder_layers)]
        references = [output_proposals.new_full((batch_size, max_queries, 4), 0.5)
                      for _ in range(reference_layers)]
        encoder_targets = output_memory.new_zeros(batch_size, max_queries, hidden_size)
        encoder_references = output_proposals.new_full((batch_size, max_queries, 4), 0.5)
        initial_boxes = output_proposals.new_zeros(batch_size, max_queries, 4)

        for (sample_indices, group_hidden, group_references,
             group_targets, group_reference_logits, group_initial_boxes) in grouped_results:
            count = group_targets.shape[1]
            for layer_index, layer_output in enumerate(group_hidden):
                hidden_states[layer_index][sample_indices, :count] = layer_output
            for layer_index, layer_reference in enumerate(group_references):
                references[layer_index][sample_indices, :count] = layer_reference
            encoder_targets[sample_indices, :count] = group_targets
            encoder_references[sample_indices, :count] = group_reference_logits.sigmoid()
            initial_boxes[sample_indices, :count] = group_initial_boxes

        query_valid_mask = (
            torch.arange(max_queries, device=query_counts.device).unsqueeze(0)
            < query_counts.unsqueeze(1))
        return (hidden_states, references, encoder_targets.unsqueeze(0),
                encoder_references.unsqueeze(0), initial_boxes, query_valid_mask)

    def forward(self, srcs, masks, pos_embeds, dn_targets, args_dn):
        # ========== 输入准备 ==========
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        enc_topk_proposals = enc_refpoint_embed = None

        # ========== Encoder ==========
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            ref_token_index=enc_topk_proposals,
            ref_token_coord=enc_refpoint_embed,
        )

        # ========== AQBA query-budget allocation ==========
        real_counts = None
        if self.training and dn_targets is not None:
            real_counts = torch.tensor([len(target['labels']) for target in dn_targets],
                                       device=memory.device, dtype=torch.float32)

        allocator_outputs = {}
        executed_query_count = self.num_queries
        query_counts = torch.full(
            (memory.shape[0],), self.num_queries, device=memory.device, dtype=torch.long)

        try:
            allocator_outputs = self.query_allocator(
                memory, spatial_shapes=spatial_shapes, real_counts=real_counts,
                epoch=self.current_epoch)

            if allocator_outputs and isinstance(allocator_outputs, dict):
                if 'boundaries' in allocator_outputs and 'log_boundaries' in allocator_outputs:
                    boundaries = allocator_outputs['boundaries']
                    log_boundaries = allocator_outputs['log_boundaries']

                    from util.box_ops import validate_boundary_predictions
                    valid_mask = validate_boundary_predictions(boundaries, log_boundaries)

                    if not valid_mask.all():
                        allocator_outputs['query_counts'] = torch.where(
                            valid_mask, allocator_outputs['query_counts'],
                            torch.full_like(allocator_outputs['query_counts'], self.num_queries))

                query_counts = allocator_outputs['query_counts'].long()
                if self.force_query_budget is not None:
                    query_counts = torch.full_like(query_counts, int(self.force_query_budget))
                max_queries = max(self.query_budget_levels)
                query_counts = query_counts.clamp(min=min(self.query_budget_levels), max=max_queries)
                executed_query_count = int(query_counts.max().item())

        except Exception as e:
            # Numeric sample-level fallbacks live in AQBA. Programming/OOM errors
            # must not silently turn a failed experiment into another architecture.
            raise RuntimeError('AQBA allocation failed') from e

        if self.two_stage_type == 'standard':
            learned_wh = self.two_stage_wh_embedding.weight[0] if self.two_stage_learn_wh else None
            proposal_valid = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, learned_wh, valid_mask_only=True)
        else:
            proposal_valid = ~mask_flatten
        valid_token_counts = proposal_valid.sum(dim=1).long()
        if (valid_token_counts == 0).any():
            raise ValueError('Each image requires at least one valid encoder proposal')
        query_counts = torch.minimum(query_counts, valid_token_counts)
        executed_query_count = int(query_counts.max().item())
        allocator_outputs['query_counts'] = query_counts
        allocator_outputs['executed_query_counts'] = query_counts
        # Supervise exactly the real encoder cells, including odd-sized inputs.
        allocator_outputs['density_valid_mask'] = ~masks[0][:, None]
        query_valid_mask = (
            torch.arange(executed_query_count, device=memory.device).unsqueeze(0) < query_counts.unsqueeze(1))

        # ========== DGFC feature calibration ==========
        if allocator_outputs and 'density_feature' in allocator_outputs:
            try:
                density_pyramid_features = self.density_pyramid(allocator_outputs['density_feature'])
                memory = self.feature_calibrator(
                    density_pyramid_features, memory, spatial_shapes)
            except Exception as e:
                raise RuntimeError('DGFC calibration failed') from e

        density_prior_flat = None
        if 'density_prior' in allocator_outputs:
            density_prior_flat = self.flatten_density_prior(
                allocator_outputs['density_prior'], spatial_shapes, mask_flatten)

        # ========== DN准备 ==========
        tgt, refpoint_embed, attn_mask, dn_meta = \
            prepare_for_cdn(dn_args=(dn_targets, args_dn[0], args_dn[1], args_dn[2]),
                            training=args_dn[3], num_queries=executed_query_count, num_classes=args_dn[4],
                            hidden_dim=args_dn[5], label_enc=args_dn[6])

        # ========== Two-stage处理 ==========
        if self.two_stage_type == 'standard':
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None

            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes,
                                                                           input_hw)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            if self.two_stage_pat_embed > 0:
                bs, nhw, _ = output_memory.shape
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(1, self.two_stage_pat_embed, 1)

            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None
                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat((output_proposals, refpoint_embed), dim=1)

            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory) + output_proposals

            can_group_decoder = (
                self.grouped_decoder_inference and not self.training
                and self.two_stage_pat_embed == 0
                and self.two_stage_add_query_num == 0
                and not self.two_stage_keep_all_tokens
                and density_prior_flat is not None)
            if can_group_decoder:
                (hs, references, hs_enc, ref_enc, init_box_proposal,
                 query_valid_mask) = self._decode_grouped_inference(
                    output_memory, output_proposals,
                    enc_outputs_class_unselected, enc_outputs_coord_unselected,
                    density_prior_flat, mask_flatten, memory,
                    lvl_pos_embed_flatten, level_start_index, spatial_shapes,
                    valid_ratios, query_counts)
                decoder_tokens = query_counts.sum()
                legacy_tokens = query_counts.max() * query_counts.numel()
                allocator_outputs['decoder_query_tokens'] = decoder_tokens
                allocator_outputs['legacy_query_tokens'] = legacy_tokens
                allocator_outputs['query_token_reduction'] = (
                    1.0 - decoder_tokens.float() / legacy_tokens.clamp(min=1).float())
                allocator_outputs['executed_query_counts'] = query_counts
                return (hs, references, hs_enc, ref_enc, init_box_proposal, dn_meta,
                        allocator_outputs, query_counts, query_valid_mask)

            topk = min(executed_query_count, enc_outputs_class_unselected.shape[1])
            if density_prior_flat is None:
                topk_proposals = torch.topk(
                    enc_outputs_class_unselected.max(-1)[0].masked_fill(
                        mask_flatten | ~torch.isfinite(output_proposals).all(-1), float('-inf')),
                    topk, dim=1).indices
            else:
                topk_proposals = self.select_proposal_indices(
                    enc_outputs_class_unselected, density_prior_flat, mask_flatten, topk,
                    proposal_boxes=output_proposals)

            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1,
                                             topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()

            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[0:topk, None, :].repeat(1, bs, 1).transpose(0, 1)
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == 'no':
            bs = memory.shape[0]
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(self.num_queries, 1)
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()
        else:
            raise NotImplementedError(f"unknown two_stage_type {self.two_stage_type}")

        # ========== Decoder ==========
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            tgt_key_padding_mask=self._build_target_padding_mask(
                tgt, query_valid_mask)
        )
        batch_max_tokens = query_counts.max() * query_counts.numel()
        allocator_outputs['decoder_query_tokens'] = batch_max_tokens
        allocator_outputs['legacy_query_tokens'] = batch_max_tokens
        allocator_outputs['query_token_reduction'] = batch_max_tokens.float() * 0.0

        # ========== Postprocess ==========
        if self.two_stage_type == 'standard':
            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals
            else:
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None

        return (hs, references, hs_enc, ref_enc, init_box_proposal, dn_meta,
                allocator_outputs, query_counts, query_valid_mask)


class TransformerEncoder(nn.Module):

    def __init__(self,
                 encoder_layer, num_layers, norm=None, d_model=256,
                 num_queries=300,
                 deformable_encoder=False,
                 enc_layer_share=False, enc_layer_dropout_prob=None,
                 two_stage_type='no',
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.two_stage_type = two_stage_type
        if two_stage_type in ['enceachlayer', 'enclayer1']:
            _proj_layer = nn.Linear(d_model, d_model)
            _norm_layer = nn.LayerNorm(d_model)
            if two_stage_type == 'enclayer1':
                self.enc_norm = nn.ModuleList([_norm_layer])
                self.enc_proj = nn.ModuleList([_proj_layer])
            else:
                self.enc_norm = nn.ModuleList([copy.deepcopy(_norm_layer) for i in range(num_layers - 1)])
                self.enc_proj = nn.ModuleList([copy.deepcopy(_proj_layer) for i in range(num_layers - 1)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device), indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                src: Tensor,
                pos: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                key_padding_mask: Tensor,
                ref_token_index: Optional[Tensor] = None,
                ref_token_coord: Optional[Tensor] = None
                ):
        if self.two_stage_type in ['no', 'standard', 'enceachlayer', 'enclayer1']:
            assert ref_token_index is None

        output = src
        if self.num_layers > 0:
            if self.deformable_encoder:
                reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        intermediate_output = []
        intermediate_ref = []
        if ref_token_index is not None:
            out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        for layer_id, layer in enumerate(self.layers):
            dropflag = False
            if self.enc_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:
                if self.deformable_encoder:
                    output = layer(src=output, pos=pos, reference_points=reference_points,
                                   spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                                   key_padding_mask=key_padding_mask)
                else:
                    output = layer(src=output.transpose(0, 1), pos=pos.transpose(0, 1),
                                   key_padding_mask=key_padding_mask).transpose(0, 1)

            if ((layer_id == 0 and self.two_stage_type in ['enceachlayer', 'enclayer1']) \
                or (self.two_stage_type == 'enceachlayer')) \
                    and (layer_id != self.num_layers - 1):
                output_memory, output_proposals = gen_encoder_output_proposals(output, key_padding_mask, spatial_shapes)
                output_memory = self.enc_norm[layer_id](self.enc_proj[layer_id](output_memory))

                topk = self.num_queries
                enc_outputs_class = self.class_embed[layer_id](output_memory)
                ref_token_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
                ref_token_coord = torch.gather(output_proposals, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, 4))

                output = output_memory

            if (layer_id != self.num_layers - 1) and ref_token_index is not None:
                out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
                intermediate_output.append(out_i)
                intermediate_ref.append(ref_token_coord)

        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:
            intermediate_output = torch.stack(intermediate_output)
            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=False,
                 num_feature_levels=1,
                 deformable_decoder=False,
                 decoder_query_perturber=None,
                 dec_layer_number=None,
                 rm_dec_query_scale=False,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 use_detached_boxes_dec_out=False
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,
                level_start_index: Optional[Tensor] = None,
                spatial_shapes: Optional[Tensor] = None,
                valid_ratios: Optional[Tensor] = None,
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] \
                                             * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points)
                reference_points_input = None

            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)

            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid()
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / reference_points[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / reference_points[..., 3]).unsqueeze(-1)

            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                output = layer(
                    tgt=output,
                    tgt_query_pos=query_pos,
                    tgt_query_sine_embed=query_sine_embed,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_reference_points=reference_points_input,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_level_start_index=level_start_index,
                    memory_spatial_shapes=spatial_shapes,
                    memory_pos=pos,
                    self_attn_mask=tgt_mask,
                    cross_attn_mask=memory_mask
                )

            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                    nq_now = new_reference_points.shape[0]
                    select_number = self.dec_layer_number[layer_id + 1]
                    if nq_now != select_number:
                        class_unselected = self.class_embed[layer_id](output)
                        topk_proposals = torch.topk(class_unselected.max(-1)[0], select_number, dim=0)[1]
                        new_reference_points = torch.gather(new_reference_points, 0,
                                                            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))

                if self.rm_detach and 'dec' in self.rm_detach:
                    reference_points = new_reference_points
                else:
                    reference_points = new_reference_points.detach()
                if self.use_detached_boxes_dec_out:
                    ref_points.append(reference_points)
                else:
                    ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))
            if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                if nq_now != select_number:
                    output = torch.gather(output, 0,
                                          topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 add_channel_attention=False,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 ):
        super().__init__()
        if use_deformable_box_attn:
            self.self_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points,
                                                      used_func=box_attn_type)
        else:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 key_aware_type=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 ):
        super().__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']
        if use_deformable_box_attn:
            self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points,
                                                       used_func=box_attn_type)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        if decoder_sa_type == 'ca_content':
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self, tgt, tgt_query_pos=None, tgt_query_sine_embed=None,
                   tgt_key_padding_mask=None, tgt_reference_points=None,
                   memory=None, memory_key_padding_mask=None,
                   memory_level_start_index=None, memory_spatial_shapes=None,
                   memory_pos=None, self_attn_mask=None, cross_attn_mask=None):
        if self.self_attn is not None:
            if self.decoder_sa_type == 'sa':
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask,
                                     key_padding_mask=tgt_key_padding_mask,
                                     need_weights=False)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_label':
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                                      tgt_reference_points.transpose(0, 1).contiguous(),
                                      memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                                      memory_key_padding_mask).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))
        return tgt

    def forward_ca(self, tgt, tgt_query_pos=None, tgt_query_sine_embed=None,
                   tgt_key_padding_mask=None, tgt_reference_points=None,
                   memory=None, memory_key_padding_mask=None,
                   memory_level_start_index=None, memory_spatial_shapes=None,
                   memory_pos=None, self_attn_mask=None, cross_attn_mask=None):
        if self.key_aware_type is not None:
            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                               memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        return tgt

    def forward(self, tgt, tgt_query_pos=None, tgt_query_sine_embed=None,
                tgt_key_padding_mask=None, tgt_reference_points=None,
                memory=None, memory_key_padding_mask=None,
                memory_level_start_index=None, memory_spatial_shapes=None,
                memory_pos=None, self_attn_mask=None, cross_attn_mask=None):
        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed,
                                      tgt_key_padding_mask, tgt_reference_points,
                                      memory, memory_key_padding_mask, memory_level_start_index,
                                      memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed,
                                      tgt_key_padding_mask, tgt_reference_points,
                                      memory, memory_key_padding_mask, memory_level_start_index,
                                      memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))
        return tgt


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from .utils import RandomBoxPerturber
        decoder_query_perturber = RandomBoxPerturber(
            x_noise_scale=args.dln_xy_noise, y_noise_scale=args.dln_xy_noise,
            w_noise_scale=args.dln_hw_noise, h_noise_scale=args.dln_hw_noise)

    use_detached_boxes_dec_out = False
    try:
        use_detached_boxes_dec_out = args.use_detached_boxes_dec_out
    except:
        use_detached_boxes_dec_out = False

    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,

        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        use_deformable_box_attn=args.use_deformable_box_attn,
        box_attn_type=args.box_attn_type,

        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,

        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,

        two_stage_type=args.two_stage_type,
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,

        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,

        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out,

        query_budget_levels=args.query_budget_levels,
        allocator_fallback_queries=args.allocator_fallback_queries,
        use_ema=args.use_ema,
        allocator_use_boundary_ema=getattr(args, 'allocator_use_boundary_ema', True),
        ema_decay=args.ema_decay,
        calibrator_gate_type=args.calibrator_gate_type,
        calibrator_use_spatial=args.calibrator_use_spatial,
        calibrator_spatial_alphas=args.calibrator_spatial_alphas,
        proposal_selection_mode=args.proposal_selection_mode,
        proposal_density_weight=args.proposal_density_weight,
        mixed_density_ratio=args.mixed_density_ratio,
        grouped_decoder_inference=args.grouped_decoder_inference,
        force_query_budget=getattr(args, 'force_query_budget', None),
    )
