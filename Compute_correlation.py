#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def compute_correlation(data = openfoodfacts, methode='Spearman'):
    """
    Compute spearman or pearson correlation
    
    Args:
        data(dataset) : A dataset with variables
        methode : correlation method
            - 'S' or Spearman to spearman correlation
            - 'P' or Pearson to pearson correlation
        
    Returns:
        A specific dataframe with signifiant correlation and associate p-value. 
        Schema :
            f1 and f2 column from data with f1 <> f2
            correlation: correlation rate
            p-value: associate p-value 
            nrows: Number of rows in commun between f1 and f2
    """
    
    # Compute Correlation 

    import scipy

    dfcorr = pd.DataFrame()
    feat1s = []
    feat2s = []
    corrs = []
    p_values = []
    nrows = []

    testCol = [x for x in data.columns if '_100g' in x]
    
    if str(methode).capitalize() in ['S','Spearman']:
        mcorr = scipy.stats.spearmanr
    else:
        mcorr = scipy.stats.pearsonr

    for feat1 in data[testCol]:
        for feat2 in data[testCol]:
            if feat1 != feat2:
                feat1s.append(feat1)
                feat2s.append(feat2)
                c1 = data[[feat1,feat2]].dropna(how='any',axis=0)[feat1]
                c2 = data[[feat1,feat2]].dropna(how='any',axis=0)[feat2]

                nrowss = data[[feat1,feat2]].dropna(how='any',axis=0).shape[0]

                if c1.shape[0]>2:
                    corr, p_value = mcorr(c1, c2)
                else:
                    corr, p_value = (0,1)

                corrs.append(corr)
                p_values.append(p_value)
                nrows.append(nrowss)

    dfcorr['Feature_1'] = feat1s
    dfcorr['Feature_2'] = feat2s
    dfcorr['Correlation'] = corrs
    dfcorr['p_value'] = p_values
    dfcorr['nrows'] = nrows

    rescorr = dfcorr.loc[dfcorr['Correlation'].notna() & dfcorr['p_value'].notna(),:]
    rescorr = rescorr[rescorr['p_value']<0.05]
    rescorr


    pd.options.display.max_rows = 200
    rescorrelationP = rescorr[(rescorr.nrows>100) & (abs(rescorr.Correlation)>0.8) ]    .groupby('Feature_1').apply(lambda x: x.sort_values(['Correlation'], ascending=False).head(3))    .rename(columns={'Feature_1':'feature_1','Feature_2':'feature_2'},)    .sort_values(['Feature_1','Correlation'], ascending=[True,False] )


    rescorrelationP['f1'] = [x if x<y else y for x,y in zip(rescorrelationP.feature_1,rescorrelationP.feature_2)]
    rescorrelationP['f2'] = [x if x>y else y for x,y in zip(rescorrelationP.feature_1,rescorrelationP.feature_2)]

    return( rescorrelationP.reset_index()    .drop(columns=['level_1','Feature_1','feature_1','feature_2'])    .drop_duplicates()[['f1','f2','Correlation','p_value','nrows']] )

