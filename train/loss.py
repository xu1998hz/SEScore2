class loss_funct():
    def __init__(self, strategy, emb_type, margin=0):
        self.strategy = strategy
        self.sim = Similarity(temp=config.temp)
        # max_length and hidden size are dataset specfic and model specfic (XLM-Roberta)
        if strategy == 'in_batch':
            self.loss_fct = nn.CrossEntropyLoss()
        elif strategy == 'mt_out_hard_one':
            self.loss_fct_1, self.loss_fct_2 = nn.MultiMarginLoss(margin=margin, reduction='mean'), nn.MultiMarginLoss(margin=margin*2, reduction='mean')
        else:
            print("We currently supports two strategies")
            exit(1)

    def loss_compute(self, train_batch, model, emb_type):
        src_pool_embed = pool(model, train_batch['input_ids'], train_batch['src_attn_masks'], emb_type)
        pos_pool_embed = pool(model, train_batch['tar'], train_batch['tar_attn_masks'], emb_type)

        if self.strategy == 'in_batch':
            cos_sim = self.sim(src_pool_embed.unsqueeze(1), pos_pool_embed.unsqueeze(0))
            # each label correspond to index of sample in the bacth, only the same index is the correct class
            labels = torch.arange(cos_sim.size(0)).long().to(config.device)
            loss = self.loss_fct(cos_sim, labels) # default is mean

        elif self.strategy == 'margin_src_ref_batch':
            pass
        elif self.strategy == 'margin_src_ref_mt1':
            pass

        elif self.strategy == 'margin_src_ref_mt1_batch':
            pass

        elif self.strategy == 'mt_out_hard_one':
            # debatable needs to further investigate
            mt_pool_embed = pool(model, train_batch['mt1'], train_batch['mt1_attn_masks'], emb_type)
            cos_sim_src_pos_batch_neg = self.sim(src_pool_embed.unsqueeze(1), pos_pool_embed.unsqueeze(0))
            cos_sim_hard_neg_batch_neg = self.sim(mt_pool_embed.unsqueeze(1), pos_pool_embed.unsqueeze(0))
            cos_sim_src_pos, cos_sim_src_hard_neg = torch.diag(cos_sim_src_pos_batch_neg, 0), self.sim(src_pool_embed, mt_pool_embed)
            cos_sim_pos_hard_neg_pair = torch.cat((cos_sim_src_pos.unsqueeze(1), cos_sim_src_hard_neg.unsqueeze(1)), dim=1)
            # pairwise ranking y labels between (src,ref,mt), (src, ref, neg), (mt, ref, neg)
            src_ref_mt_y = torch.tensor([0]*cos_sim_pos_hard_neg_pair.size(0)).to(config.device) # margin * 1
            src_ref_neg_y = torch.arange(cos_sim_src_pos_batch_neg.size(0)).to(config.device) # margin * 2
            mt_ref_neg_y = torch.arange(cos_sim_hard_neg_batch_neg.size(0)).to(config.device) # margin * 1
            # compute loss
            src_ref_mt_loss = self.loss_fct_1(cos_sim_pos_hard_neg_pair, src_ref_mt_y)
            src_ref_neg_loss = self.loss_fct_2(cos_sim_src_pos_batch_neg, src_ref_neg_y)
            mt_ref_neg_loss = self.loss_fct_1(cos_sim_hard_neg_batch_neg, mt_ref_neg_y)
            loss = 1/3*(src_ref_mt_loss + src_ref_neg_loss + mt_ref_neg_loss)
        return loss
