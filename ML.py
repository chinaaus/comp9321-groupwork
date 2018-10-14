import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle


def data_predict(df, inputlist):
    x_train = df.drop(['App','Rating','Reviews','Installs','Last Updated','Current Ver','Android Ver'], axis=1).values
    y_train = df['Installs'].values
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    predictions = knn.predict(inputlist)
    return predictions


if __name__ == '__main__':
    csv_file = 'test.csv'
    df = pd.read_csv(csv_file)
    #this part should tranfer the string in the dataframe into number
    
    
    #input the required values from GUI by API
    Category = 1
    Size = 23
    Type = 1
    Price = 2.3
    Content = 1
    Genres = 4
    #form an input list
    inputlist = [[Category, Size, Type, Price, Content, Genres]]
    #input the dataframe and input list to the function
    #the output is the prediction
    predict_install = data_predict(df, inputlist)[0]
    
    print(predict_install)


