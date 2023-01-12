import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self,user_num,item_num,factors,layers,dropout,model,GMF_model=None,MLP_model=None):
        super(NCF,self).__init__()

        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factors)
        self.embed_item_GMF = nn.Embedding(item_num, factors)

        self.embed_user_MLP = nn.Embedding(user_num,factors*(2**(layers-1)))
        self.embed_item_MLP= nn.Embedding(item_num,factors*(2**(layers-1)))


        MLP_block = []
        for i in range(layers):
            input_size = factors * (2 ** (layers - i))
            MLP_block.append(nn.Dropout(self.dropout))
            MLP_block.append(nn.Linear(input_size,input_size//2))
            MLP_block.append(nn.ReLU())

        self.MLP_layers = nn.Sequential(*MLP_block)

        if self.model in ['MLP','GMF']:
            self.final_layer = nn.Linear(factors,1)
        
        else:
            self.final_layer = nn.Linear(2*factors,1)

        self._init_weight_()

    def _init_weight_(self):
        """Initialize model parameter"""
        if self.model == 'NeuMF-pre':
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)

            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

            for m1,m2 in self.MLP_layers, self.MLP_model.MLP_layers:
                if isinstance(m1,nn.Linear) and isinstance(m2,nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            
            alpha = 0.5
            predict_weight = torch.cat([alpha*self.GMF_model.final_layer.weight, (1-alpha)*self.MLP_model.final_layer.weight],dim=1)
            predict_bias = self.GMF_model.final_layer.bias + self.MLP_model.final_layer.bias

            self.final_layer.weight.data.copy_(predict_weight)
            self.final_layer.bias.data.copy_(predict_bias)

        else:
            nn.init.normal_(self.embed_user_GMF.weight,std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight,std=0.01)

            nn.init.normal_(self.embed_user_MLP.weight,std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight,std=0.01)

            for m in self.MLP_layers:
                if isinstance(m,nn.Linear):
                    nn.init.normal_(m.weight,std=0.01)
            
            nn.init.normal_(self.final_layer.weight,std=0.01)

    
    def forward(self,user,item):
        if self.model == 'MLP':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)

            MLP_layer_input = torch.cat((embed_user_MLP,embed_item_MLP),-1)
            output_MLP = self.MLP_layers(MLP_layer_input)

            score = self.final_layer(output_MLP)

            return score.view(-1)
        
        if self.model == 'GMF':
            embed_item_GMF = self.embed_item_GMF(item)
            embed_user_GMF = self.embed_user_GMF(user)
            output_GMF = embed_user_GMF * embed_item_GMF

            score = self.final_layer(output_GMF)

            return score.view(-1)

        if self.model == 'NeuMF-pre' or 'NeuMF-end':
            #MLP part
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)

            MLP_layer_input = torch.cat((embed_user_MLP,embed_item_MLP),-1)
            output_MLP = self.MLP_layers(MLP_layer_input)


            #GMF part
            embed_item_GMF = self.embed_item_GMF(item)
            embed_user_GMF = self.embed_user_GMF(user)
            output_GMF = embed_user_GMF * embed_item_GMF


            #prediction part
            output_NeuMF = torch.cat((output_MLP,output_GMF),-1)
            score = self.final_layer(output_NeuMF)

            return score.view(-1)