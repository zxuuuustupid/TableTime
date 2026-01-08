import pandas as pd
import json
import torch

class ts2DFLoader(torch.nn.Module):
    def __init__(self,channel_list,n_sample,frequency,time_use):
        super(ts2DFLoader,self).__init__()
        self.channel_list=channel_list
        time_stamps = [pd.Timedelta(seconds=i / frequency) for i in range(n_sample)]
        time_stamps = [str(t.total_seconds()) + 's' for t in time_stamps]
        self.time_stamps=time_stamps
        self.time_use=time_use
    def forward(self,x):
        data = x.T
        column_names = self.channel_list
        df = pd.DataFrame(data, columns=column_names)
        DFLoader=dict()
        for i in range(len(self.channel_list)):
            DFLoader[column_names[i]]=list(df[column_names[i]])
        DFLoader=str(DFLoader)
        if self.time_use==False:
            result = 'pd.DataFrame(' + DFLoader + f',index={df.index.values.tolist()}' + ')'
            return result
        else:
            result = 'pd.DataFrame(' + DFLoader + f',index={self.time_stamps}' + ')'
            return result
          
class ts2html(torch.nn.Module):
    def __init__(self,channel_list,n_sample,frequency,time_use):
        super(ts2html,self).__init__()
        time_stamps = [pd.Timedelta(seconds=i / frequency) for i in range(n_sample)]
        time_stamps = [str(t.total_seconds())+'s' for t in time_stamps]
        self.time_stamps = time_stamps
        self.time_use = time_use
        self.channel_list=channel_list
    def forward(self,x):
        data = x.T
        column_names = self.channel_list
        if self.time_use==True:
            df = pd.DataFrame(data, columns=column_names, index=self.time_stamps)
            result = df.to_html(index=True).replace(' ', '')
            return result
        else:
            df = pd.DataFrame(data, columns=column_names)
            result = df.to_html(index=True).replace(' ', '')
            return result
          
class ts2json(torch.nn.Module):
    def __init__(self,channel_list,n_sample,frequency,time_use):
        super(ts2json,self).__init__()

        time_stamps = [pd.Timedelta(seconds=i / frequency) for i in range(n_sample)]
        time_stamps = [str(t.total_seconds())+'s' for t in time_stamps]
        self.time_stamps = time_stamps
        self.time_use = time_use
        self.channel_list=channel_list
    def forward(self,x):
        data=x.T
        column_names=self.channel_list

        if self.time_use==True:
            df=pd.DataFrame(data,columns=column_names,index=self.time_stamps)
            data = {str(index): row.to_dict() for index, row in df.iterrows()}
            return json.dumps(data, indent=4, ensure_ascii=False).replace(' ','')
        else:
            df = pd.DataFrame(data, columns=column_names)
            data = {str(index): row.to_dict() for index, row in df.iterrows()}
            return json.dumps(data, indent=4, ensure_ascii=False).replace(' ', '')
          
class ts2markdown(torch.nn.Module):
    def __init__(self, channel_list,n_sample,frequency,time_use):
        super(ts2markdown, self).__init__()
        time_stamps = [pd.Timedelta(seconds=i / frequency) for i in range(n_sample)]
        time_stamps = [str(t.total_seconds()) + 's' for t in time_stamps]
        self.time_stamps = time_stamps
        self.time_use = time_use
        self.channel_list = channel_list
    def forward(self, x):
        data = x.T
        column_names=self.channel_list
        if self.time_use == True:
            df = pd.DataFrame(data, columns=column_names, index=self.time_stamps)
            df = df.to_markdown(index=True).replace(' ','')
            return df
        else:
            df = pd.DataFrame(data, columns=column_names)
            df = df.to_markdown(index=True).replace(' ','')
            return df
