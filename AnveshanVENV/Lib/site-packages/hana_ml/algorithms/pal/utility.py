"""
This module contains Python API of utility functions.
"""
import json
import logging
import os
import re
import pandas as pd
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
from hana_ml.dataframe import create_dataframe_from_pandas
from hana_ml.algorithms.pal.partition import train_test_val_split

#pylint: disable=bare-except, line-too-long, invalid-name, no-member, no-self-use, unspecified-encoding
#pylint: disable=consider-using-f-string
def version_compare(pkg_version, version):
    """
    If pkg's version is greater than the specified version, it returns True. Otherwise, it returns False.
    """
    pkg_ver_list = pkg_version.split(".")
    ver_list = version.split(".")
    if int(pkg_ver_list[0]) > int(ver_list[0]):
        return True
    if int(pkg_ver_list[0]) == int(ver_list[0]):
        if int(pkg_ver_list[1]) > int(ver_list[1]):
            return True
        if int(pkg_ver_list[1]) == int(ver_list[1]):
            if int(pkg_ver_list[2]) >= int(ver_list[2]):
                return True
    return False

def check_pal_function_exist(connection_context, func_name, like=False):
    """
    Check the existence of pal function.
    """
    operator = '='
    if like:
        operator = 'like'
    exist = connection_context.sql('SELECT * FROM "SYS"."AFL_FUNCTIONS" WHERE AREA_NAME = \'AFLPAL\' and FUNCTION_NAME {} \'{}\';'.format(operator, func_name))
    if len(exist.collect()) > 0:
        return True
    return False

class AMDPHelper(object):
    """
    AMDP Generation helper.
    """
    def __init__(self):
        self.amdp_template_replace = {}
        self.amdp_template = ''
        self.fit_data = None
        self.predict_data = None
        self.abap_class_mapping_dict = {}
        self.label = None

    def add_amdp_template(self, template_name):
        """
        Add AMDP template
        """
        self.amdp_template = self.load_amdp_template(template_name)

    def add_amdp_name(self, amdp_name):
        """
        Add AMDP name.
        """
        self.amdp_template_replace["<<AMDP_NAME>>"] = amdp_name

    def add_amdp_item(self, template_key, value):
        """
        Add item.
        """
        self.amdp_template_replace[template_key] = value

    def build_amdp_class(self):
        """
        After add_item, generate amdp file from template.
        """
        for key, val in self.amdp_template_replace.items():
            self.amdp_template = self.amdp_template.replace(key, val)

    def write_amdp_file(self, filepath=None, version=1, outdir="out"):
        """
        Write template to file.
        """
        if filepath:
            with open(filepath, "w+") as file:
                file.write(self.amdp_template)
        else:
            create_dir = os.path.join(outdir,
                                      self.amdp_template_replace["<<AMDP_NAME>>"],
                                      "abap")
            os.makedirs(create_dir, exist_ok=True)
            filename = "Z_CL_{}_{}.abap".format(self.amdp_template_replace["<<AMDP_NAME>>"], version)
            with open(os.path.join(create_dir, filename), "w+") as file:
                file.write(self.amdp_template)

    def get_amdp_notfillin_key(self):
        """
        Get AMDP not fillin keys.
        """
        return re.findall("(<<[a-zA-Z_]+>>)", self.amdp_template)

    def load_amdp_template(self, template_name):
        """
        Load AMDP template
        """
        filepath = os.path.join(os.path.dirname(__file__),
                                "..",
                                "..",
                                "artifacts",
                                "generators",
                                "filewriter",
                                "templates",
                                template_name)
        with open(filepath, 'r') as file:
            return file.read()

    def load_abap_class_mapping(self):
        """
        Load ABAP class mapping.
        """
        filepath = os.path.join(os.path.dirname(__file__),
                                "..",
                                "..",
                                "artifacts",
                                "config",
                                "data",
                                "hdbtable_to_abap_datatype_mapping.json")
        with open(filepath, 'r') as file:
            self.abap_class_mapping_dict = json.load(file)

    def abap_class_mapping(self, value):
        """
        Mapping the abap class.
        """
        if 'VARCHAR' in value.upper():
            if 'NVARCHAR' in value.upper():
                return self.abap_class_mapping_dict['NVARCHAR']
            else:
                return self.abap_class_mapping_dict['VARCHAR']
        if 'DECIMAL' in value.upper():
            return self.abap_class_mapping_dict['DECIMAL']
        return self.abap_class_mapping_dict[value]


class Settings:
    """
    Configuration of logging level
    """
    settings = None
    user = None
    @staticmethod
    def load_config(config_file):
        """
        Load HANA credentials.
        """
        Settings.settings = configparser.ConfigParser()
        Settings.settings.read(config_file)
        try:
            url = Settings.settings.get("hana", "url")
        except:
            url = ""
        try:
            port = Settings.settings.getint("hana", "port")
        except:
            port = 0
        try:
            pwd = Settings.settings.get("hana", "passwd")
        except:
            pwd = ''
        try:
            Settings.user = Settings.settings.get("hana", "user")
        except:
            Settings.user = ""
        Settings._init_logger()
        return url, port, Settings.user, pwd

    @staticmethod
    def _set_log_level(logger, level):
        if level == 'info':
            logger.setLevel(logging.INFO)
        else:
            if level == 'warn':
                logger.setLevel(logging.WARN)
            else:
                if level == 'debug':
                    logger.setLevel(logging.DEBUG)
                else:
                    logger.setLevel(logging.ERROR)

    @staticmethod
    def _init_logger():
        logging.basicConfig()
        for module in ["hana_ml.ml_base", 'hana_ml.dataframe', 'hana_ml.algorithms.pal']:
            try:
                level = Settings.settings.get("logging", module)
            except:
                level = "error"
            logger = logging.getLogger(module)
            Settings._set_log_level(logger, level.lower())

    @staticmethod
    def set_log_level(level='info'):
        """
        Set logging level.

        Parameters
        ----------

        level : {'info', 'warn', 'debug', 'error'}
        """
        logging.basicConfig()
        for module in ["hana_ml.ml_base", 'hana_ml.dataframe', 'hana_ml.algorithms.pal']:
            logger = logging.getLogger(module)
            Settings._set_log_level(logger, level)

class DataSets:
    """
    Load demo data.
    """
    @staticmethod
    def load_bank_data(connection,
                       schema=None,
                       chunk_size=10000,
                       force=False,
                       train_percentage=.50,
                       valid_percentage=.40,
                       test_percentage=.10,
                       full_tbl="DBM2_RFULL_TBL",
                       seed=1234,
                       url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/bank-additional-full.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url,
                                          header=None,
                                          names=['AGE',
                                                 'JOB',
                                                 'MARITAL',
                                                 'EDUCATION',
                                                 'DBM_DEFAULT',
                                                 'HOUSING',
                                                 'LOAN',
                                                 'CONTACT',
                                                 'DBM_MONTH',
                                                 'DAY_OF_WEEK',
                                                 'DURATION',
                                                 'CAMPAIGN',
                                                 'PDAYS',
                                                 'PREVIOUS',
                                                 'POUTCOME',
                                                 'EMP_VAR_RATE',
                                                 'CONS_PRICE_IDX',
                                                 'CONS_CONF_IDX',
                                                 'EURIBOR3M',
                                                 'NREMPLOYED',
                                                 'LABEL'])
            data.insert(0, "ID", range(0, len(data)))
            data.set_index("ID")
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        train_df, test_df, valid_df = train_test_val_split(full_df,
                                                           id_column="ID",
                                                           random_seed=seed,
                                                           partition_method='random',
                                                           training_percentage=train_percentage,
                                                           testing_percentage=test_percentage,
                                                           validation_percentage=valid_percentage)
        return full_df, train_df, valid_df, test_df

    @staticmethod
    def load_titanic_data(connection,
                          schema=None,
                          chunk_size=10000,
                          force=False,
                          train_percentage=.50,
                          valid_percentage=.40,
                          test_percentage=.10,
                          full_tbl="TITANIC_FULL_TBL",
                          seed=1234,
                          url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/titanic-full.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url,
                                          header=None,
                                          names=['PASSENGER_ID',
                                                 'PCLASS',
                                                 'NAME',
                                                 'SEX',
                                                 'AGE',
                                                 'SIBSP',
                                                 'PARCH',
                                                 'TICKET',
                                                 'FARE',
                                                 'CABIN',
                                                 'EMBARKED',
                                                 'SURVIVED'])
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        train_df, test_df, valid_df = train_test_val_split(full_df,
                                                           id_column="PASSENGER_ID",
                                                           random_seed=seed,
                                                           partition_method='random',
                                                           training_percentage=train_percentage,
                                                           testing_percentage=test_percentage,
                                                           validation_percentage=valid_percentage)
        return full_df, train_df, valid_df, test_df

    @staticmethod
    def load_walmart_data(connection,
                          schema=None,
                          chunk_size=10000,
                          force=False,
                          full_tbl="WALMART_TRAIN_TBL",
                          url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/walmart-train.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url,
                                          header=None,
                                          names=['ITEM_IDENTIFIER',
                                                 'ITEM_WEIGHT',
                                                 'ITEM_FAT_CONTENT',
                                                 'ITEM_VISIBILITY',
                                                 'ITEM_TYPE',
                                                 'ITEM_MRP',
                                                 'OUTLET_IDENTIFIER',
                                                 'OUTLET_ESTABLISHMENT_YEAR',
                                                 'OUTLET_SIZE',
                                                 'OUTLET_LOCATION_IDENTIFIER',
                                                 'OUTLET_TYPE',
                                                 'ITEM_OUTLET_SALES'])
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        return full_df

    @staticmethod
    def load_iris_data(connection,
                       schema=None,
                       chunk_size=10000,
                       force=False,
                       train_percentage=.50,
                       valid_percentage=.40,
                       test_percentage=.10,
                       full_tbl="IRIS_DATA_FULL_TBL",
                       seed=1234,
                       url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/iris.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url,
                                          header=None,
                                          names=['SEPALLENGTHCM',
                                                 'SEPALWIDTHCM',
                                                 'PETALLENGTHCM',
                                                 'PETALWIDTHCM',
                                                 'SPECIES'])
            data.insert(0, "ID", range(0, len(data)))
            data.set_index("ID")
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        train_df, test_df, valid_df = train_test_val_split(full_df,
                                                           id_column="ID",
                                                           random_seed=seed,
                                                           partition_method='random',
                                                           training_percentage=train_percentage,
                                                           testing_percentage=test_percentage,
                                                           validation_percentage=valid_percentage)
        return full_df, train_df, valid_df, test_df

    @staticmethod
    def load_boston_housing_data(connection,
                                 schema=None,
                                 chunk_size=10000,
                                 force=False,
                                 train_percentage=.50,
                                 valid_percentage=.40,
                                 test_percentage=.10,
                                 full_tbl="BOSTON_HOUSING_PRICES",
                                 seed=1234,
                                 url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/boston-house-prices.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url,
                                          header=None,
                                          names=["CRIM",
                                                 "ZN",
                                                 "INDUS",
                                                 "CHAS",
                                                 "NOX",
                                                 "RM",
                                                 "AGE",
                                                 "DIS",
                                                 "RAD",
                                                 "TAX",
                                                 "PTRATIO",
                                                 "BLACK",
                                                 "LSTAT",
                                                 "MEDV",
                                                 "ID"])
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        train_df, test_df, valid_df = train_test_val_split(full_df,
                                                           id_column="ID",
                                                           random_seed=seed,
                                                           partition_method='random',
                                                           training_percentage=train_percentage,
                                                           testing_percentage=test_percentage,
                                                           validation_percentage=valid_percentage)
        return full_df, train_df, valid_df, test_df

    @staticmethod
    def load_flight_data(connection,
                         schema=None,
                         chunk_size=10000,
                         force=False,
                         train_percentage=.50,
                         valid_percentage=.40,
                         test_percentage=.10,
                         full_tbl="FLIGHT_DATA_FULL_TBL",
                         seed=1234,
                         url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/flight.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url,
                                          header=None,
                                          names=['YEAR',
                                                 'MONTH',
                                                 'DAY',
                                                 'DAY_OF_WEEK',
                                                 'AIRLINE',
                                                 'FLIGHT_NUMBER',
                                                 'TAIL_NUMBER',
                                                 'ORIGIN_AIRPORT',
                                                 'DESTINATION_AIRPORT',
                                                 'SCHEDULED_DEPARTURE',
                                                 'DEPARTURE_TIME',
                                                 'DEPARTURE_DELAY',
                                                 'TAXI_OUT',
                                                 'WHEELS_OFF',
                                                 'SCHEDULED_TIME',
                                                 'ELAPSED_TIME',
                                                 'AIR_TIME',
                                                 'DISTANCE',
                                                 'WHEELS_ON',
                                                 'TAXI_IN',
                                                 'SCHEDULED_ARRIVAL',
                                                 'ARRIVAL_TIME',
                                                 'ARRIVAL_DELAY',
                                                 'DIVERTED',
                                                 'CANCELLED',
                                                 'CANCELLATION_REASON',
                                                 'AIR_SYSTEM_DELAY',
                                                 'SECURITY_DELAY',
                                                 'AIRLINE_DELAY',
                                                 'LATE_AIRCRAFT_DELAY',
                                                 'WEATHER_DELAY'])
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        train_df, test_df, valid_df = train_test_val_split(full_df.add_id("ID"),
                                                           id_column="ID",
                                                           random_seed=seed,
                                                           partition_method='random',
                                                           training_percentage=train_percentage,
                                                           testing_percentage=test_percentage,
                                                           validation_percentage=valid_percentage)
        return full_df.deselect("ID"), train_df.deselect("ID"), valid_df.deselect("ID"), test_df.deselect("ID")

    @staticmethod
    def load_adult_data(connection,
                        schema=None,
                        chunk_size=10000,
                        force=False,
                        train_percentage=.50,
                        valid_percentage=.40,
                        test_percentage=.10,
                        full_tbl="ADULT_DATA_FULL_TBL",
                        seed=1234,
                        url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/adult.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url,
                                          header=None,
                                          names=['AGE',
                                                 'WORKCLASS',
                                                 'FNLWGT',
                                                 'EDUCATION',
                                                 'EDUCATIONNUM',
                                                 'MARITALSTATUS',
                                                 'OCCUPATION',
                                                 'RELATIONSHIP',
                                                 'RACE',
                                                 'SEX',
                                                 'CAPITALGAIN',
                                                 'CAPITALLOSS',
                                                 'HOURSPERWEEK',
                                                 'NATIVECOUNTRY',
                                                 'INCOME'])
            data.insert(0, "ID", range(0, len(data)))
            data.set_index("ID")
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        train_df, test_df, valid_df = train_test_val_split(full_df,
                                                           id_column="ID",
                                                           random_seed=seed,
                                                           partition_method='random',
                                                           training_percentage=train_percentage,
                                                           testing_percentage=test_percentage,
                                                           validation_percentage=valid_percentage)
        return full_df, train_df, valid_df, test_df

    @staticmethod
    def load_diabetes_data(connection,
                           schema=None,
                           chunk_size=10000,
                           force=False,
                           train_percentage=.50,
                           valid_percentage=.40,
                           test_percentage=.10,
                           full_tbl="PIMA_INDIANS_DIABETES_TBL",
                           seed=1234,
                           url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/pima-indians-diabetes.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url,
                                          header=None,
                                          names=['PREGNANCIES',
                                                 'GLUCOSE',
                                                 'SKINTHICKNESS',
                                                 'INSULIN',
                                                 'BMI',
                                                 'AGE',
                                                 'CLASS'])
            data.insert(0, "ID", range(0, len(data)))
            data.set_index("ID")
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        train_df, test_df, valid_df = train_test_val_split(full_df,
                                                           id_column="ID",
                                                           random_seed=seed,
                                                           partition_method='random',
                                                           training_percentage=train_percentage,
                                                           testing_percentage=test_percentage,
                                                           validation_percentage=valid_percentage)
        return full_df, train_df, valid_df, test_df

    @staticmethod
    def load_shampoo_data(connection,
                          schema=None,
                          chunk_size=10000,
                          force=False,
                          full_tbl="SHAMPOO_SALES_DATA_TBL",
                          url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/shampoo.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url,
                                          header=None,
                                          names=['ID',
                                                 'SALES'])
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        return full_df

    @staticmethod
    def load_apriori_data(connection,
                          schema=None,
                          chunk_size=10000,
                          force=False,
                          full_tbl="PAL_APRIORI_TRANS_TBL",
                          url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/apriori_item_data.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url,
                                          header=None,
                                          names=['CUSTOMER',
                                                 'ITEM'])
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        return full_df


    @staticmethod
    def load_spm_data(connection,
                      schema=None,
                      chunk_size=10000,
                      force=False,
                      full_tbl="PAL_SPM_DATA_TBL",
                      url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/spm_data.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url,
                                          header=None,
                                          names=['CUSTID',
                                                 'TRANSID',
                                                 'ITEMS'])
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        return full_df

    @staticmethod
    def load_covid_data(connection,
                        schema=None,
                        chunk_size=10000,
                        force=False,
                        full_tbl="PAL_COVID_DATA_TBL",
                        url="https://raw.githubusercontent.com/SAP-samples/hana-ml-samples/main/Python-API/pal/datasets/worldwide-aggregated.csv"):
        """
        Load data.
        """
        if schema is None:
            schema = connection.get_current_schema()
        if connection.has_table(full_tbl, schema) and not force:
            print("Table {} exists.".format(full_tbl))
            full_df = connection.table(full_tbl, schema)
        else:
            data = pd.io.parsers.read_csv(url)
            full_df = create_dataframe_from_pandas(connection,
                                                   pandas_df=data,
                                                   table_name=full_tbl,
                                                   schema=schema,
                                                   force=force,
                                                   chunk_size=chunk_size)
        return full_df
