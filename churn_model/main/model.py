import pandas as pd
import verticapy as vp
import qwak
from verticapy.utilities import *
from verticapy.learn.tools import *
from qwak.model.base import QwakModelInterface
from verticapy.learn.ensemble import RandomForestClassifier
from verticapy.learn.preprocessing import OneHotEncoder

# create connection with vertica DB
CONN_INFO = {
            "host": "140.236.88.151",
            "port": "5433",
            "database": "PartPub80DB",
            "password": "qwak_23",
            "user": "qwak_user",
        }
vp.new_connection(CONN_INFO,name='qwak_conn')


class ChurnClassifier(QwakModelInterface):
    """ The Model class inherit QwakModelInterface base class
    """
    def __init__(self):
        # Connect with the DB
        vp.connect('qwak_conn')
        # Initialize the model
        self.rf_model = RandomForestClassifier(name="qwak_schema.rf_churn",
                                               n_estimators=10,
                                               max_features="auto",
                                               max_leaf_nodes=32,
                                               sample=0.7,
                                               max_depth=5,
                                               min_samples_leaf=1,
                                               min_info_gain=0.0,
                                               nbins=32)

    @staticmethod
    def data_preparation(vdf, train = True):
        """method to prepare data"""

        vdf["TotalCharges"].dropna()
        for column in [
            "DeviceProtection",
            "MultipleLines",
            "PaperlessBilling",
            "TechSupport",
            "Partner",
            "StreamingTV",
            "OnlineBackup",
            "Dependents",
            "OnlineSecurity",
            "PhoneService",
            "StreamingMovies",
            ]:
            vdf[column].decode("Yes", 1, 0)
        if train:
            enc = OneHotEncoder(name = "qwak_schema.ohe_churn")

            enc.fit(vdf,
                    ["gender", "Contract", "PaymentMethod", "InternetService"])

            vdf = enc.transform(vdf,
                                    ["gender", "Contract", "PaymentMethod", "InternetService"])
        else:
            model_enc = load_model("qwak_schema.ohe_churn")
            vdf = model_enc.transform(vdf,
                                    ["gender", "Contract", "PaymentMethod", "InternetService"])
            
        vdf.drop(["gender", "Contract", "PaymentMethod", "InternetService"])
        return vdf

    def build(self):
        """ 
        Responsible for loading the model. This method is invoked during build time (qwak build command)
        This function train and save the model inside vertica using data inside vertica
        """
        # Read the data from DB and prepare it for model training
        churn = ChurnClassifier.data_preparation(vp.vDataFrame("qwak_schema.churn"))
        churn['churn'].decode("Yes", 1, 0)
        # Split the data to train and test
        train, test = churn.train_test_split(test_size=0.2, random_state=0)

        # drop the model if already exist
        self.rf_model.drop()
        # features
        predictors = train.get_columns(exclude_columns=['churn', 'customerID'])
        # fit the model
        self.rf_model.fit(input_relation=train,
                          X=predictors,
                          y='churn',
                          test_relation=test)
        # Compute the metric score
        self.rf_model.predict(test, predictors, 'churn_pred')
        f1_score = test.score(y_true="churn", y_score="churn_pred", method="f1")
        accuracy_score = test.score(y_true="churn", y_score="churn_pred", method="accuracy")
        qwak.log_metric({"accuracy_score": accuracy_score})
        qwak.log_metric({"f1_score": f1_score})
        

    @qwak.api()
    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """ Invoked on every API inference request.
        Args:
            input_data (Dataframe): datframe with one colum with table name to use for prediction 

        Returns: model output (inference results), as datafram wit the size of the input and f1 and accuracy
                    metrics to make sure that the prediction is preformed
        """
        # get the table name from the input
        table_name = input_data['table_name'].values[0]
        # read the table
        input_data = vp.vDataFrame(table_name, schema="qwak_schema")
        # prepare the dat for prediction
        input_data = ChurnClassifier.data_preparation(input_data, train=False)
        input_data['churn'].decode("Yes", 1, 0)
        # list of the independent variables
        predictors = input_data.get_columns(exclude_columns=['churn', 'customerID'])
        # predict
        output_data = self.rf_model.predict(input_data, predictors, name='churn_pred')
        # save the results in DB
        output_data.to_db(name='"qwak_schema"."churn_ref"', relation_type="insert")
        # compute metrics
        f1_score = output_data.score(y_true="churn", y_score="churn_pred", method="f1")
        accuracy_score = output_data.score(y_true="churn", y_score="churn_pred", method="accuracy")
        length = len(output_data['churn_pred'])
        # create Dataframe form meterics
        res = pd.DataFrame(index=[0], data={"size":length,
                                            "f1" : f1_score,
                                            "accuracy":accuracy_score})
        qwak.log_metric({"accuracy_score": accuracy_score})
        qwak.log_metric({"f1score": f1_score})
        return res

