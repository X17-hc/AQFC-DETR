_base_ = ['../aitodv2/aqfc_r50_5scale_24e.py']
force_query_budget = 900
allocator_fallback_queries = 900
proposal_selection_mode = 'fused'
grouped_decoder_inference = False
