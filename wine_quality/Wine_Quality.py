import pickle 

class WineQuality(object):
    def __init__(self):
        self.free_sulfer_scaler = pickle.load(open('parameter/free_sulfur_scaler.pkl', 'rb'))
        self.total_sulfur_dioxide = pickle.load(open('parameter/total_sulfur_dioxide.pkl', 'rb'))
        
    def data_preparation(self, df):
        #rescaling free
        df['free sulfur dioxide'] = self.free_sulfer_scaler.transform(df[['free sulfur dioxide']])
        #rescaling total sulfur
        df['total sulfur dioxide'] = self.total_sulfur_dioxide.transform(df[['total sulfur dioxide']])
        
        return df