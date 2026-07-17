import torch
import torch.nn as nn 
import torch.nn.functional as F
import math

class SDALayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def scaled_dot_product(self, q, k, v, attn_mask, key_padding_mask):
        d_k = q.size()[-1]
        # print(q.shape, k.transpose(-2, -1).shape, k.shape)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k) # [20, 4, 64, 64]
        
        # print("attn:", attn_logits.shape)
        if not key_padding_mask is None:
            # k = k * (1 - key_padding_mask # = [20,1,64,1] 
            # assert False, key_padding_mask.shape
            # print(attn_logits.shape, key_padding_mask.shape)
            attn_logits = attn_logits.masked_fill(key_padding_mask,float('-inf'))
            
        ## masks: [20, 64], q/k/v: [20, 4, 64, 32]
        if not attn_mask is None:
            # attn_logits[]
            # attn_logits[:,:,]
            assert False
            
        attention = F.softmax(attn_logits, dim=-1)
        
        # print("before final matmul:",attention.shape, v.shape)
        values = torch.matmul(attention, v)
        return values, attn_logits

    def forward(self, q, k, v, attn_mask, key_padding_mask) -> torch.Tensor:
        return self.scaled_dot_product(q,k,v, attn_mask, key_padding_mask)


class ConvAttention(nn.Module):
    def __init__(self, input_dim, att_stride=2, att_kernel=3, head_dim_mul=0.5,fuseconv=False, querykey=False, convs=True, retAtt=False, reduc_channel=1.0):
        super().__init__()
      
        self.head_dim_mul = head_dim_mul 
        self.num_heads = int(max(1,(input_dim*self.head_dim_mul*reduc_channel)//30))
        self.input_dim = input_dim
        self.head_dim = int((input_dim*reduc_channel // self.num_heads) * self.head_dim_mul)
        self.num_keys = 2 if querykey else 3
        self.att_stride = att_stride
        self.querykey = querykey
        self.convs = convs
        self.retAtt = retAtt
        
        total_dim = int(self.head_dim * self.num_heads * self.num_keys)
        
        if self.convs:
            self.conv_proj = nn.Sequential(nn.Conv2d(input_dim,input_dim, kernel_size=3, stride=att_stride, padding=1, groups=input_dim), nn.BatchNorm2d(input_dim) )
        
        self.pwise = nn.Sequential(nn.Conv2d(input_dim, total_dim, kernel_size=1, stride=1, padding=0, bias=False))
        
        
        self.sda = SDALayer()

        
        self.o_proj_inpdim = self.head_dim * self.num_heads
        
        # self.o_proj = nn.Linear(self.o_proj_inpdim, input_dim)
        if not querykey:
            
            self.o_proj = nn.Conv2d(self.o_proj_inpdim, int(input_dim*reduc_channel), kernel_size=1,stride=1,padding=0)

            
            self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=att_stride*2, stride=att_stride, padding=att_stride//2  , groups=input_dim)
            if att_stride == 1 :
                self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim)
        
        
            if fuseconv:
                self.o_proj = nn.Identity()
                if att_stride == 1 :
                    self.upsampling = nn.ConvTranspose2d(self.o_proj_inpdim, input_dim,kernel_size=3, stride=1, padding=1)
                else:
                    self.upsampling = nn.ConvTranspose2d(self.o_proj_inpdim, input_dim, kernel_size=att_stride*2, stride=att_stride, padding=att_stride//2  )
            if not self.convs:
                self.upsampling = nn.Identity()
                
        if querykey:
            self.query_linear_input = nn.Linear(input_dim, int(input_dim * self.head_dim_mul))
            self.query_linear_out = nn.Linear(int(input_dim * self.head_dim_mul), input_dim)
            

    # def forward_query(self, ):
        
    
    def forward_cross(self, query, keyvalue, attn_mask=None, key_padding_mask=None):
        N, C, H, W = keyvalue.size()

        ## Conv + proj : of keyvalue Image features
        if self.convs:
            keyvalue = self.conv_proj(keyvalue)
        xout = self.pwise(keyvalue)
        ## Proj of query vectors
        query_proj = self.query_linear_input(query).reshape(N,-1, self.num_heads, self.head_dim)
        # -> [15,N,4,32]
        query_proj = query_proj.permute(0,2,1,3) # -> [N,4,15,32]
        
        
        ## Split image features into K,V 
        N, c, h, w = xout.size()
        # Separate Q, K, V from linear output
        kv = xout.reshape(N, self.num_heads, self.num_keys*self.head_dim, h*w)
        kv = kv.permute(0, 1, 3, 2) # [N, Head, SeqLen, Dims]
        k, v = kv.chunk(2, dim=-1)         

        ## Attention
        # masks: [20, 256]
        # query: [N, L, DIM] , key/value [20, 4, 64, 32]
        values, attn_logits = self.sda(query_proj, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # [N,4,15,32] -> [N,L,Head,Dim]
        
        ## Reshape query vectors
        values = values.permute(0,2,1,3).reshape(N, -1, self.num_heads*self.head_dim)

        ## Proj query vectors
        outquery = self.query_linear_out(values) # -> [15,N, 256]

        if self.retAtt:
            return outquery, attn_logits
        return outquery
        
    def forward(self, x, mean_query=False, attn_mask=None, key_padding_mask=None):
            
        N, C, H, W = x.size()
        if self.convs:
            x = self.conv_proj(x)
        xout = self.pwise(x)
        
        # Transformer part
        N, c, h, w = xout.size()
        # assert H / self.att_stride == h, "H:%d h:%f H/h:%f" %(H,h,H/h)
        # print(xout.shape, self.num_heads, self.num_keys, self.head_dim)
        # Separate Q, K, V from linear output
        qkv = xout.reshape(N, self.num_heads, self.num_keys*self.head_dim, h*w)

        
        
        qkv = qkv.permute(0, 1, 3, 2) # [N, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)         

        if mean_query:
            q = q.mean(dim=2,keepdim=True)
        # masks: [20, 256]
        values, _ = self.sda(q,k,v, attn_mask=attn_mask, key_padding_mask=key_padding_mask) # -> 
    
        if mean_query:
            o = self.o_proj(values.permute(0,1,3,2).reshape(N,self.o_proj_inpdim,1,1)) #[N,C,h,w]
            return o.reshape(N,-1)
        else:
            o = self.o_proj(values.permute(0,1,3,2).reshape(N,self.o_proj_inpdim,h,w)) #[N,C,h,w]

        # Upsampling
        o = self.upsampling(o)
        
        return o[:N,:C,:H,:W]