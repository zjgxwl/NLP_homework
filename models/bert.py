# coding: UTF-8
import torch
import torch.nn as nn
import copy
from pytorch_pretrained import BertModel, BertTokenizer
import torch.nn.functional as F

# from transformers import BertModel, BertTokenizer

class Prompt(nn.Module):
    def __init__(self, embed_dim=768, embedding_key='cls', prompt_init='zero', prompt_pool=True, 
                 prompt_key=True, pool_size=None, top_k=2, batchwise_prompt=False, prompt_key_init='normal', num_classes=10, device=None, num_per_class=4):
        super().__init__()

        ## 40 个pool
        ## 每个类别4个， 每个任务8个
        ## topk 4

        self.compress_dim = 768
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.sig = nn.Sigmoid()
        self.device = device
        self.num_per_class = num_per_class
        # self.fc = nn.Linear(embed_dim, self.compress_dim)

        # self.id_dict = {0 : torch.arange(0,2,1), 1 : torch.arange(2,4,1), 2 : torch.arange(4,6,1), 3 : torch.arange(6,8,1), 4 : torch.arange(8,10,1), 
        #                 5 : torch.arange(10,12,1), 6 : torch.arange(12,14,1), 7 : torch.arange(14,16,1), 8 : torch.arange(16,18,1), 9 : torch.arange(18,20,1)}
        
        self.id_dict = dict()
        # self.id_dict = {0 : torch.arange(0,4,1), 1 : torch.arange(4,8,1), 2 : torch.arange(8,12,1), 3 : torch.arange(12,16,1), 4 : torch.arange(16,20,1), 
        #                 5 : torch.arange(20,24,1), 6 : torch.arange(24,28,1), 7 : torch.arange(28,32,1), 8 : torch.arange(32,36,1), 9 : torch.arange(36,40,1)}

        for i in range(num_classes):
            self.id_dict[i] = torch.arange(i*self.num_per_class, (i+1)*self.num_per_class, 1)
        
        if self.prompt_pool:
            prompt_pool_shape = (pool_size, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
            elif prompt_init == 'normal':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.normal_(self.prompt, 0.035, 0.01)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, self.compress_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, 0, 1)
            elif prompt_key_init == 'normal':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.normal_(self.prompt_key, 0.035, 0.01)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
        
    
    def forward(self, x_embed=None, prompt_mask=None, cls_features=None, mode='train', labels=None):
        out = dict()
        # if mode == 'test':
        #     self.top_k = 2
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            # if mode == 'train':
            #     similarity = similarity * (1/self.prompt_count)

            if prompt_mask is None:
            
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k

                out['idx_learned'] = copy.deepcopy(idx)

                # if mode=='train':
                #     for i in range(len(torch.bincount(idx_1D))):
                #         self.prompt_count[i] = self.prompt_count[i] + torch.bincount(idx_1D)[i]
                # if self.batchwise_prompt:
                #     prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                #     # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                #     # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                #     # Unless dimension is specified, this will be flattend if it is not already 1D.
                #     if prompt_id.shape[0] < self.pool_size:
                #         prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                #         id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                #     _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                #     major_prompt_id = prompt_id[major_idx] # top_k
                #     # expand to batch
                #     idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k
            similarity_nagative = None

            if labels != None:
                # task_id = int(labels[0]) // 2
                # print(task_ids)
                for i in range(labels.shape[0]):
                    # print(int(task_ids[i]))
                    # idx[i] = self.id_dict[int(labels[i])]
                    
                    # 根据label 取出对应索引
                    # 取出 similarity 对应的行和列
                    # softmax 计算权重， 采样得到最终索引，记得转换device
                    
                    classes_candiate = self.id_dict[int(labels[i])]
                    similarity_candiate = similarity[i,classes_candiate]
                    soft_weights = F.softmax(similarity_candiate * 0.5, dim=0)
                    idx[i] = torch.multinomial(soft_weights, self.top_k, replacement=False)+self.num_per_class*int(labels[i])
                    idx[i] = idx[i].to(self.device)
                   
                # similarity_nagative = torch.matmul(x_embed_norm, prompt_norm[(task_id+1)*8:,:].t()) # B, Pool_size
               
                # similarity_positive = similarity[idx]
           
            # if task_id != -1:
            #     idx = self.id_dict[task_id]
            #     idx = idx.expand(cls_features.shape[0], -1)
            
            batched_prompt_raw = self.prompt[idx] # B, top_k
            batch_size, top_k, _ = batched_prompt_raw.shape
            # batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C
            batched_prompt = torch.mean(batched_prompt_raw, dim=1)
            batched_prompt = batched_prompt + cls_features
            out['prompt_idx'] = idx
            
            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity


            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / sim.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim
            # if labels != None:
            #     out['similarity_nagative'] = similarity_nagative
            #     out['sim_nag'] = torch.sum(similarity_nagative) / similarity_nagative.shape[0] 
                # out['similarity_positive'] = similarity_positive
                # out['sim_pos'] = torch.sum(similarity_positive) / similarity_positive.shape[0] 
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = batched_prompt
        # if mode == 'test':
        #     self.top_k = 4
        return out


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        # self.train_path = dataset + '/data/train.txt'                                # 训练集
        # self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        # self.test_path = dataset + '/data/test.txt'                                  # 测试集
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 15                                           # 类别数
        self.num_epochs = 3                                             # epoch数
        # self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 0.03                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.memory_size = 0
        self.top_k = 2
        self.num_per_class = 8
        self.pool_size = self.num_per_class * self.num_classes


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.bert = BertModel.from_pretrained(config.bert_path)
   
        
        self.prompt = Prompt(embed_dim=config.hidden_size, pool_size=config.pool_size, top_k=config.top_k, num_classes=config.num_classes, device=config.device, num_per_class=config.num_per_class)
        
        self.memory_size = config.memory_size
        self.data_memory = []
        
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    
    def forward(self, x, mode='train', labels=None):
        ##   不止训练prompt
          
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled_cls, pooled_mean = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # _, pooled_cls, pooled_mean = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        
        prompt_out = self.prompt(cls_features=pooled_cls, mode=mode, labels=labels)


        # prompt_out['logits'] = self.fc(pooled_cls+pooled_mean)
        prompt_out['logits'] = self.fc(prompt_out['prompted_embedding'])
        return prompt_out
    
    def update_memory(self, task_id, train_set):
        
        nums_per_class = self.memory_size // (task_id+1) // 2
        
        data_memory = []
        for class_id in range((task_id+1)*2):
            
            indexs = self.select_indexs_from_class(train_set, class_id, nums_per_class)
            # index 
            
            for i in indexs:
                data_memory.append(train_set[i])    
            
        self.data_memory = data_memory
        pass
        
    def select_indexs_from_class(self, train_set=None, class_id=0, nums_per_class=10):
        
        indexs = []
        indexs_all = []
        
        for i, item in enumerate(train_set):
            if item[1] == class_id:
                indexs_all.append(i)
        
        if nums_per_class<len(indexs_all):
            return indexs_all[:nums_per_class]
        else:
            return indexs_all  


