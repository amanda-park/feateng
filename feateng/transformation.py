import pandas as pd
import numpy as np
from scipy import stats
import scipy.linalg as la
import utils
import ppscore as pps


class Transformation:
    def __init__(self,df):
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
        #core attributes
        self._df = df
        self._ls_hist = [df.copy()]
        self._ls_steps = []
        #optional attributes
        self._ls_mean = []
        self._ls_std = []
        self._ls_max = []
        self._ls_min = []
        self._ls_lamb = []
        self._ls_feature_vecs = []
        self._ls_features = []
         
    def step_rm(self):
        #if the step is standard_normal(), then remove optional attributes
        if self._ls_steps[-1] == 'standard_normal()':
            del self._ls_mean[-1]
            del self._ls_std[-1]
        #if the step is normalize(), then remove optional attributes
        if self._ls_steps[-1] == 'normalize()':
            del self._ls_max[-1]
            del self._ls_min[-1]
        #if the step is box_cox(), then remove optional attributes
        elif self._ls_steps[-1] == 'box_cox()':
            del self._ls_lamb[-1]   
        #if the step is box_cox(), then remove optional attributes
        elif self._ls_steps[-1] == 'pca()':
            del self._ls_feature_vecs[-1]   
        #if the step is box_cox(), then remove optional attributes
        elif self._ls_steps[-1] == 'feature_select()':
            del self._ls_features[-1]   
        #remove and reset core attributes
        del self._ls_steps[-1]
        del self._ls_hist[-1]
        self._df = self._ls_hist[-1]
        
    def standard_normal(self):
        #append mean and std step to optional attributes
        trans_df = self._df.copy()
        self._ls_mean.append(trans_df.mean(axis=0))
        self._ls_std.append(trans_df.std(axis=0)) 
        #append and update core attributes
        trans_df = trans_df.apply(lambda x: (x - self._ls_mean[-1])/self._ls_std[-1], axis=1)
        for i in trans_df.columns:
            self._df[i] = trans_df[i]
        self._ls_hist.append(self._df.copy())
        self._ls_steps.append('standard_normal()')
        
    def normalize(self):
        #append mean and std step to optional attributes

        trans_df = self._df.copy()
        self._ls_max.append(trans_df.max(axis=0))
        self._ls_min.append(trans_df.min(axis=0)) 
        #append and update core attributes
        trans_df = trans_df.apply(lambda x: (x - self._ls_min[-1])/(self._ls_max[-1] - self._ls_min[-1]), axis=1)
        for i in trans_df.columns:
            self._df[i] = trans_df[i]
        self._ls_hist.append(self._df.copy())
        self._ls_steps.append('normalize()')
    
    def box_cox(self,drop_cols = None):
        if drop_cols == None:
            float_df = self._df.copy()
        else:
            #drop selected columns
            float_df = self._df.drop(drop_cols,axis=1)
        #keep all float columns
        float_df = float_df.select_dtypes(include=['float64','float32'])
        #keep all strictly positive columns
        for i in float_df.columns:
            TF = (float_df[i] < 0).any().any()
            if TF == True:
                float_df = float_df.drop(i,axis=1)
        #calculate p-values for normality test on original dataframe
        p_val_orig = float_df.apply(lambda x: stats.normaltest(x), axis=0).iloc[1:]
        #create transformation dataframe
        trans_df = float_df.copy()
        for i in float_df.columns:
            trans_df[i]=utils.bc_df(float_df[i])
        #create lambda dataframe
        lamb_df = p_val_orig.copy()
        for i in float_df.columns:
            lamb_df[i]=utils.bc_lambda(float_df[i])
        #calculate p-values for normality test on transformed dataframe
        p_val_trans = trans_df.apply(lambda x: stats.normaltest(x), axis=0).iloc[1:]
        #keep columns in trans_df and lamb_df that improve the normality of variables
        for i in float_df.columns:
            TF = p_val_orig[i].iloc[0] > p_val_trans[i].iloc[0]
            if TF:
                trans_df = trans_df.drop(i,axis=1)
                lamb_df = lamb_df.drop(i,axis=1)
        #append lambda step to optional attributes
        self._ls_lamb.append(lamb_df)
        #append and update core attributes
        for i in trans_df.columns:
            self._df[i] = trans_df[i]
        self._ls_hist.append(self._df.copy())
        self._ls_steps.append('box_cox()')
        
    def pca(self):
        trans_df = self._df.copy()
        cov = trans_df.cov().to_numpy()
        eigvals, eigvecs = la.eig(cov)
        
        feature_vecs = np.transpose(eigvecs)
        trans_np = trans_df.to_numpy()
        trans_np = np.transpose(trans_np)
        pca_np = np.matmul(feature_vecs, trans_np)
        self._ls_feature_vecs.append(feature_vecs)
        pca_cols = []
        for i in range(len(self._df.columns)):
            pca_cols.append('pc_'+str(i+1))       
        self._df = pd.DataFrame(np.transpose(pca_np), columns = pca_cols)
        self._ls_hist.append(self._df.copy())
        self._ls_steps.append('pca()')        
        
    def feature_select(self, y_df, method='ppscore', cutoff = .1):
        trans_df = self._df.copy()
        if method == 'ppscore':
            trans_df['y'] = y_df
            outcome = pps.predictors(trans_df,'y')
            outcome_x = outcome.copy()
            outcome_x[['ppscore']] = outcome_x[outcome_x[['ppscore']]>=cutoff][['ppscore']]
            outcome_x = outcome_x.dropna()
            outcome_x = outcome_x.x
        else:
            #'pearson','spearman','kendall'
            y_df = y_df.iloc[:,0]
            outcome = trans_df.corrwith(y_df,method=method,axis=0)
            outcome_x = outcome[abs(outcome)>=cutoff].index
        self._df = trans_df[outcome_x]
        self._ls_features.append(outcome_x)
        self._ls_hist.append(self._df.copy())
        self._ls_steps.append('feature_select()')
        return outcome
    
    def transform_new(self,new_df):
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
        j,k,l,m,n = 0,0,0,0,0
        for i in range(len(self._ls_steps)):
            step = self._ls_steps[i]
            if step == 'standard_normal()':
                temp_df = new_df.copy()
                temp_df = temp_df.apply(lambda x: (x - self._ls_mean[j])/self._ls_std[j], axis=1)
                for i in temp_df.columns:
                    new_df[i] = temp_df[i]
                j+=1
            elif step == 'normalize()':
                temp_df = new_df.copy()
                temp_df = temp_df.apply(lambda x: (x - self._ls_min[k])/(self._ls_max[k] - self._ls_min[k]), axis=1)
                for i in temp_df.columns:
                    new_df[i] = temp_df[i]
                k+=1
            elif step == 'box_cox()':
                temp_df = new_df.copy()
                lamb_df = self._ls_lamb[l]
                temp_df = temp_df[lamb_df.columns]
                for column in temp_df.columns:
                    lamb = lamb_df[column].iloc[0]
                    if lamb == 0:
                        new_df[column] = temp_df[column].apply(lambda x: np.log(x))
                    else:
                        new_df[column] = temp_df[column].apply(lambda x: ((x**lamb)-1)/lamb)
                l+=1
            elif step == 'pca()':
                temp_df - new_df.copy()
                trans_np = temp_df.to_numpy()
                trans_np = np.transpose(trans_np)
                feature_vecs = self._ls_feature_vecs[m]
                pca_np = np.matmul(feature_vecs, trans_np)
                pca_cols = []
                for i in range(len(temp_df.columns)):
                    pca_cols.append('pc_'+str(i+1))       
                new_df = pd.DataFrame(np.transpose(pca_np), columns = pca_cols)
                m+=1
            elif step == 'feature_select()':
                col_select = self._ls_features[n]
                new_df = new_df[col_select]
                n+=1
        return new_df
        
if __name__ == '__main__':
    df_x = pd.read_csv('train_x.csv')
    df_y = pd.read_csv('train_y.csv')
    df_x = df_x.to_numpy()
    Y = df_x[:, 0]
    X = df_x[:, 1:]
    print(Y,X)
    print(df_y)
    #out = Transformation(df_x)
    #out.box_cox()
    #out.standard_normal()
    #out.normalize()
    #out.pca()
    #out.standard_normal()
    #out.normalize()
    #out.feature_select(y_df=df_y,method='kendall',cutoff=.08)
    #df_test = pd.read_csv('test.csv')
    #print(out._df)
    #print(out.transform_new(df_test))
