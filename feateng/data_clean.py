import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from utils import tukeys_method


class Data_Clean:

    """This class allows manipulating pandas dataframes to format data appropriately for analysis.
    """    

    def __init__(self, df):
        self._df = df

    def convert_vartypes(self, num_to_cat_cutoff=0):
        """Modifies the data frame fed into DATA_CLEAN and converts variable types to the appropriate type needed for further analysis.
        
        :param int num_to_cat_cutoff: Optional; default is 0.
            Will force a numeric variable to be categorical if the frequency of values is below the cutoff. If zero, no numeric variables will be converted to categorical.

        :returns: None
        """

        # Check numeric variables
        num_varnames = self._df.select_dtypes(
            include=["float64", "int64", "int", "float", "int32", "float32", "double"]).columns.tolist()
        # If numeric values are not very frequent, convert to categorical
        for i in range(0, len(num_varnames)):
            if (len(self._df[num_varnames[i]].unique()) <= num_to_cat_cutoff):
                self._df[num_varnames[i]] = pd.Categorical(self._df[num_varnames[i]])
            else:
                self._df[num_varnames[i]] = pd.to_numeric(self._df[num_varnames[i]])

        # Make sure all objects are cast as type categorical
        obj_varnames = self._df.select_dtypes(include=["object"]).columns.tolist()
        for i in range(0, len(obj_varnames)):
            self._df[obj_varnames[i]] = pd.Categorical(self._df[obj_varnames[i]])

    def reduce_levels(self, level_cutoff_percent = .01, other_name = "Other"):   
        """Modifies the data frame fed into the DATA_ClEAN class to reduce the levels of infrequently occurring categorical variables.
        
        :param float level_cutoff_percent: Optional; defaults to .01
            Will force categorical levels with a frequency below this percent to be labeled "Other". 

        :param str other_name: Optional; defaults to "Other"
            The name to give the infrequent levels of a categorical variable.

        :returns: None
        """

        cat_varnames = self._df.select_dtypes(include=["category"]).columns.tolist()
        for i in range(0, len(cat_varnames)):
            tab = pd.crosstab(index=self._df[cat_varnames[i]], columns="percent")
            pct_tab = tab / tab.sum()

            # Stores list of infrequent levels of a variable
            replace = []

            for j in range(0, len(pct_tab)):
                pct_tab_row = pct_tab[j:(j + 1)]  # Search only that specific row in the crosstab
                if ((pct_tab_row['percent'] < level_cutoff_percent).bool()):  # If frequency is less than .01
                    replace.append(pct_tab_row.index.values[0])

            def replace_with_other(x):
                if x in replace:
                    return other_name
                else:
                    return x

            self._df[cat_varnames[i]] = self._df[cat_varnames[i]].apply(replace_with_other)

    def date_col_as_index(self, date_col):
        
        """Converts the date column specified in the function into a date and turns it into an index for the pandas data frame.
        
        :param date date_col: Required; no default given
             Column to be made into the index. Will be converted to datetype if not already in that form.
            
        :returns: None
        """
        
        self._df[date_col] = pd.to_datetime(self._df[date_col])
        self._df = self._df.set_index(date_col)

    def na_cutoff(self, na_cutoff_percent = .3):
        
        """Observes all columns in the data frame and removes those with missing values higher than the parameter specified.

        :param float na_cutoff_percent: Optional; default .3
             Will force variables with NA values higher than this percentage to be removed from the dataset.

        :returns: None
        """
        
        varnames = self._df.columns.tolist()
        # Compute NA percentages for each variable (numeric and categorical)
        na_percent = self._df.isnull().sum() / self._df.shape[0]  # NAs in variable over total rows
        for i in range(0, self._df.shape[1]):  # For all variables in the dataframe
            if (na_percent[varnames[i]] > na_cutoff_percent):  # If NA Percentage > na_cutoff_percent
                self._df = self._df.drop([varnames[i]], axis=1)  # Drop variable from data frame

    def cat_ohe(self):
    
        """All categorical variables will be one-hot encoded. Useful for neural networks.
        
        :returns: None
        """    
    
        cat_varnames = self._df.select_dtypes(include=["category"]).columns.tolist()

        for i in range(0, len(cat_varnames)):
            self._df = pd.get_dummies(self._df, columns=[cat_varnames[i]], drop_first=False)

    def plot_numvars(self):
        """Plots numeric variables in histogram for quick and dirty EDA analysis.
        
        :returns: pandas histogram object
        """   
        num_varnames = self._df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        num_plt = self._df[num_varnames].hist(bins=10, figsize=(15, 6), layout=(2, 4))

        return num_plt

    def plot_catvars(self):
        """Plots categorical variables in bar plot for quick and dirty EDA analysis.
        
        :returns: pandas barplot object
        """  
        cat_varnames = self._df.select_dtypes(include=["category"]).columns.tolist()
        for i in range(0, len(cat_varnames)):
            cat_plt = self._df[cat_varnames[i]].value_counts().plot(kind='bar')

        return cat_plt

    def outliers(self):
        """Uses Tukey's Method to identify potential outliers in dataset.
        
        :returns: data frame of probable outliers, data frame of possible outliers.
        """ 
        num_varnames = self._df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        for i in range(0, len(num_varnames)):
            probable_outlier, possible_outlier = tukeys_method(self._df, num_varnames[i])

        return probable_outlier, possible_outlier


if __name__ == '__main__':
    df = pd.read_excel("DummyData.xlsx", index_col=None)
    dc = Data_Clean(df)  # Call class as dc
    dc.convert_vartypes(0)
    dc.reduce_levels(other_name = "LeadershipPreferredOther")
    dc.date_col_as_index("PredictorDate")
    dc.na_cutoff()
    dc.cat_ohe()
    clean_df = dc._df
    clean_df