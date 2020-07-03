import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn import LSTM, Linear


class LATTE(nn.Module):
    def __init__(self, args, pretrained):
        super(LATTE, self).__init__()
        self.args = args

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Sequential(
            nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width)),
            nn.ReLU()
            )

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # highway network
        assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=args.hidden_size * 2,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=args.dropout)

        # 4. Attention Flow Layer
        self.att_weight_c  = Linear(args.hidden_size * 2, 1)
        self.att_weight_q  = Linear(args.hidden_size * 2, 1)
        self.att_weight_cq = Linear(args.hidden_size * 2, 1)

        #-------------------------------------------------------------------------------------
        # 5. feed_forward_network
        self.fc1  = Linear(8800, 200)
        self.fc2  = Linear(8800, 200) #11x800
        self.relu = nn.ReLU()

        # 6. Distribution Similarity
        self.fc4  = Linear(2200, 2048)  #11*200 up uc concate k=2048
        self.fc3  = Linear(200, 2048)   #
        self.cosSi= nn.CosineSimilarity(dim=0, eps=1e-6)       #dim 尺寸确定一下

        # 7. Known Type Classifier
        self.fc5  = Linear(2048, 3)  ####???
        self.fc6  = Linear(2048, 3)

        # 8. ranking score layer
        self.f_weight = Linear(200,1)
        self.g_weight = Linear(2048,1)
        #-------------------------------------------------------------------------------------

    def forward(self, batch):
        # TODO: More memory-efficient architecture
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = F.dropout(self.char_emb(x))
            # (batch， seq_len, char_dim, word_len)
            x = x.transpose(2, 3)
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.args.char_dim, x.size(3)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze()
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args.char_channel_size)

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #cq_tiled = c_tiled * q_tiled
            #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        # 5. feed_forward_network
        #-------------------------------------------------------------------------
        def feed_forward_network(x):
            '''

            :param x: (batch, c_len, hidden_size * 2)
            :return:
            '''
            hidden1 = self.fc1(x)
            x       = self.relu(hidden1)
            # hidden2 = self.fc2(hidden1)
            # x       = self.relu(hidden2)
            return x

        # 6. Distribution Similarity
        def latent_type_similarity(Uc,Up):
            Vp      = self.fc3(Up)
            Vc      = self.fc4(Uc)
            Vp_hat  = F.softmax(Vp, dim=1)
            Vc_hat  = F.softmax(Vc, dim=1)
            g_latent= self.cosSi(Vp_hat, Vc_hat)
            return g_latent, Vp, Vc

        # 7. Known Type Classifier
        def known_type_classifier(Vp, Vc):
            yp      = self.fc5(Vp)
            yp      = self.relu(yp)
            yc      = self.fc6(Vc)
            yc      = self.relu(yc)
            return yp, yc

        # 8. Ranking Layer
        # def ranking_layer(f_feed, g_latent):
        #     r_score = torch.sum(f_feed * self.f_weight) + torch.sum(g_latent * self.g_weight)
        #     return r_score

        def ranking_layer(f_feed, g_latent):
            r_score = self.f_weight(f_feed) + self.g_weight(g_latent).mean()
            return r_score

        #-------------------------------------------------------------------------

        # 1. Character Embedding Layer
        c_char = char_emb_layer(batch.c_char)
        p_char = char_emb_layer(batch.p_char)
        n_char = char_emb_layer(batch.n_char)
        # 2. Word Embedding Layer
        c_word = self.word_emb(batch.c_word[0])
        p_word = self.word_emb(batch.p_word[0])
        n_word = self.word_emb(batch.n_word[0])
        c_lens = batch.c_word[1]
        p_lens = batch.p_word[1]
        n_lens = batch.n_word[1]

        # Highway network
        c = highway_network(c_char, c_word)
        p = highway_network(p_char, p_word)
        n = highway_network(n_char, n_word)
        # 3. Contextual Embedding Layer
        c = self.context_LSTM((c, c_lens))[0]
        p = self.context_LSTM((p, p_lens))[0]
        n = self.context_LSTM((n, n_lens))[0]
        # 4. Attention Flow Layer
        g_p = att_flow_layer(c, p).view(40, -1)  #batch size = 40
        g_n = att_flow_layer(c, n).view(40, -1)


        #---------------------------------------------------
        # 5. feed_forward_network
        f_feed_p   = feed_forward_network(g_p)
        f_feed_n   = feed_forward_network(g_n)

        # 6. Distribution Similarity
        g_latent_p, Vp_p, Vc_p = latent_type_similarity(c.view(40, -1),p.view(40, -1))
        g_latent_n, Vp_n, Vc_n = latent_type_similarity(c.view(40, -1),n.view(40, -1))
        # 7. Known Type Classifier
        yp_p, yc_p = known_type_classifier(Vp_p, Vc_p)
        yp_n, yc_n = known_type_classifier(Vp_n, Vc_n)

        # 8. Ranking Layer
        r_score_p= ranking_layer(f_feed_p, g_latent_p)
        r_score_n= ranking_layer(f_feed_n, g_latent_n)

        # (batch, c_len), (batch, c_len)
        return r_score_p, r_score_n, yp_p, yc_p, yp_n, yc_n
        #---------------------------------------------------


