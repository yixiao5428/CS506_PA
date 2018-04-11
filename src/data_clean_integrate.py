import os
import pandas as pd
import re
import csv
import xlrd


def output_temperature_production_csv(state, county, station):
    """

    :param state:
    :param county:
    :param station:
    :return:
    """
    headers = ['state', 'county', 'station', 'year',
               'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
               'plant_acres', 'harvest_acres', 'yield_acres', 'bales']

    result = {
        'state': '',
        'county': '',
        'station': '',
        'year': '',
        'jan': '',
        'feb': '',
        'mar': '',
        'apr': '',
        'may': '',
        'jun': '',
        'jul': '',
        'aug': '',
        'sep': '',
        'oct': '',
        'nov': '',
        'dec': '',
        'plant_acres': '',
        'harvest_acres': '',
        'yield_acres': '',
        'bales': ''
    }

    production_path = '../data_sets/cotton_production_data/' + state + '-' + county + '.xla'
    with open(production_path, 'r') as p_f:
        p_lines = p_f.readlines()
        for p_line in p_lines:
            tmp_list = p_line.split('\t')

            result['county'] = county
            result['state'] = state
            result['station'] = station

            if tmp_list[0].isdigit():
                result['year'] = tmp_list[0]
                result['plant_acres'] = tmp_list[1]
                result['harvest_acres'] = tmp_list[2]
                result['yield_acres'] = tmp_list[3]
                result['bales'] = tmp_list[4]

            temperature_path = '../data_sets/temperature_data/FLs.52g/' + station + '.FLs.52g.tavg'
            with open(temperature_path, 'r') as t_f:
                t_lines = t_f.readlines()
                for t_line in t_lines:
                    tmp_list = t_line.split(' ')[:-1]

                    while '' in tmp_list:
                        tmp_list.remove('')
                    while 'X' in tmp_list:
                        tmp_list.remove('X')
                    # print(tmp_list)

                    year = tmp_list[1]
                    if year == result['year']:
                        result['jan'] = re.findall('\-\d+|\d+', tmp_list[2])[0]
                        result['feb'] = re.findall('\-\d+|\d+', tmp_list[3])[0]
                        result['mar'] = re.findall('\-\d+|\d+', tmp_list[4])[0]
                        result['apr'] = re.findall('\-\d+|\d+', tmp_list[5])[0]
                        result['may'] = re.findall('\-\d+|\d+', tmp_list[6])[0]
                        result['jun'] = re.findall('\-\d+|\d+', tmp_list[7])[0]
                        result['jul'] = re.findall('\-\d+|\d+', tmp_list[8])[0]
                        result['aug'] = re.findall('\-\d+|\d+', tmp_list[9])[0]
                        result['sep'] = re.findall('\-\d+|\d+', tmp_list[10])[0]
                        result['oct'] = re.findall('\-\d+|\d+', tmp_list[11])[0]
                        result['nov'] = re.findall('\-\d+|\d+', tmp_list[12])[0]
                        result['dec'] = re.findall('\-\d+|\d+', tmp_list[13])[0]

                        print(result)
                        with open('./temperature_production.csv', 'a+', newline='') as output_f:
                            writer = csv.DictWriter(output_f, headers)
                            writer.writerow(result)

    return None


def temperature_data_pre_process():
    """

    :return:
    """
    df = pd.read_csv('../data_sets/temperature_data/station_to_county.csv')
    # print(df)

    headers = ['state', 'county', 'station', 'year',
               'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
               'plant_acres', 'harvest_acres', 'yield_acres', 'bales']
    with open('./temperature_production.csv', 'w') as output_f:
        writer = csv.DictWriter(output_f, headers)
        writer.writeheader()

    files = os.listdir('../data_sets/cotton_production_data')
    for file in files:
        try:
            c_s = file.split('.')[0].split('-')
            state = c_s[0]
            county = c_s[1]
            station_list = df[(df.county == county) & (df.state == state)].station.tolist()
            # print(station_list)

            if len(station_list) != 0:
                for station in station_list:

                    temperature_file_path = '../data_sets/temperature_data/FLs.52g/' + station + '.FLs.52g.tavg'
                    try:
                        with open(temperature_file_path, 'r') as f:
                            output_temperature_production_csv(state, county, station)
                            break

                    except FileNotFoundError as e:
                        # print(temperature_file_path)
                        continue

        except IndexError as e:
            continue
            # print(e)

    # print(files)


def output_precipitation_production_csv(state, county, station):
    """

    :param state:
    :param county:
    :param station:
    :return:
    """
    headers = ['state', 'county', 'station', 'year',
               'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
               'plant_acres', 'harvest_acres', 'yield_acres', 'bales']

    result = {
        'state': '',
        'county': '',
        'station': '',
        'year': '',
        'jan': '',
        'feb': '',
        'mar': '',
        'apr': '',
        'may': '',
        'jun': '',
        'jul': '',
        'aug': '',
        'sep': '',
        'oct': '',
        'nov': '',
        'dec': '',
        'plant_acres': '',
        'harvest_acres': '',
        'yield_acres': '',
        'bales': ''
    }

    production_path = '../data_sets/cotton_production_data/' + state + '-' + county + '.xla'
    with open(production_path, 'r') as p_f:
        p_lines = p_f.readlines()
        for p_line in p_lines:
            tmp_list = p_line.split('\t')

            result['county'] = county
            result['state'] = state
            result['station'] = station

            if tmp_list[0].isdigit():
                result['year'] = tmp_list[0]
                result['plant_acres'] = tmp_list[1]
                result['harvest_acres'] = tmp_list[2]
                result['yield_acres'] = tmp_list[3]
                result['bales'] = tmp_list[4]

            precipitation_path = '../data_sets/precipitation_data/ushcn.v2.5.5.20180319/' + station + '.raw.prcp'
            with open(precipitation_path, 'r') as t_f:
                t_lines = t_f.readlines()
                for t_line in t_lines:
                    tmp_list = t_line.split(' ')[:-1]

                    while '' in tmp_list:
                        tmp_list.remove('')
                    while 'X' in tmp_list:
                        tmp_list.remove('X')
                    # print(tmp_list)

                    year = tmp_list[1]
                    if year == result['year']:
                        result['jan'] = re.findall('\-\d+|\d+', tmp_list[2])[0]
                        result['feb'] = re.findall('\-\d+|\d+', tmp_list[3])[0]
                        result['mar'] = re.findall('\-\d+|\d+', tmp_list[4])[0]
                        result['apr'] = re.findall('\-\d+|\d+', tmp_list[5])[0]
                        result['may'] = re.findall('\-\d+|\d+', tmp_list[6])[0]
                        result['jun'] = re.findall('\-\d+|\d+', tmp_list[7])[0]
                        result['jul'] = re.findall('\-\d+|\d+', tmp_list[8])[0]
                        result['aug'] = re.findall('\-\d+|\d+', tmp_list[9])[0]
                        result['sep'] = re.findall('\-\d+|\d+', tmp_list[10])[0]
                        result['oct'] = re.findall('\-\d+|\d+', tmp_list[11])[0]
                        result['nov'] = re.findall('\-\d+|\d+', tmp_list[12])[0]
                        result['dec'] = re.findall('\-\d+|\d+', tmp_list[13])[0]

                        print(result)
                        with open('./precipitation_production.csv', 'a+', newline='') as output_f:
                            writer = csv.DictWriter(output_f, headers)
                            writer.writerow(result)

    return None


def precipitation_data_pre_process():
    """

    :return:
    """
    df = pd.read_csv('../data_sets/precipitation_data/ushcn-v2.5-stations.csv')
    # print(df)

    headers = ['state', 'county', 'station', 'year',
               'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
               'plant_acres', 'harvest_acres', 'yield_acres', 'bales']
    with open('./precipitation_production.csv', 'w') as output_f:
        writer = csv.DictWriter(output_f, headers)
        writer.writeheader()

    files = os.listdir('../data_sets/cotton_production_data/')
    for file in files:
        try:
            c_s = file.split('.')[0].split('-')
            state = c_s[0]
            county = c_s[1]
            station_list = df[(df.county == county) & (df.state == state)].station.tolist()
            # print(station_list)

            if len(station_list) != 0:
                for station in station_list:

                    precipitation_file_path = '../data_sets/precipitation_data/ushcn.v2.5.5.20180319/' + station + '.raw.prcp'
                    try:
                        with open(precipitation_file_path, 'r') as f:
                            output_precipitation_production_csv(state, county, station)
                            break

                    except FileNotFoundError as e:
                        # print(temperature_file_path)
                        continue

        except IndexError as e:
            continue
            # print(e)

    # print(files)


def merge_precipitation_temperature_production_data():
    """

    :return:
    """
    p_df = pd.read_csv('../data_sets/precipitation_production.csv')
    t_df = pd.read_csv('../data_sets/temperature_production.csv')

    df = pd.merge(p_df, t_df, how='inner',
                  on=['state', 'county', 'year', 'plant_acres', 'harvest_acres', 'yield_acres', 'bales'])

    df.to_csv('../data_sets/precipitation_temperature_production.csv', index=None)


def disease_xls_to_csv():
    """

    :return:
    """
    raw_table = xlrd.open_workbook('../data_sets/disease_data/Disease-Database.xls').sheets()[0]

    headers = ['state', 'year', 'total_percent_lost']
    with open('../data_sets/disease_data/disease_data.csv', 'w') as f:
        writer = csv.DictWriter(f, headers)
        writer.writeheader()

    result = {
        'state': '',
        'year': '',
        'total_percent_lost': ''
    }

    head_list = ['', '', 'Lost', 'AL', 'AZ', 'AR', 'CA', 'FL', 'GA', 'LA', 'MS', 'MO', 'NM', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'Bales Lost', 'Lost', '', '']

    for i in range(raw_table.nrows):
        row_list = raw_table.row_values(i)
        # print(row_list)
        if row_list[1] == 'Total Percent Lost':
            for j in range(3, 19):
                result['state'] = head_list[j]
                result['year'] = int(row_list[0])
                result['total_percent_lost'] = row_list[j]
                print(result)
                with open('../data_sets/disease_data/disease_data.csv', 'a+', newline='') as f:
                    writer = csv.DictWriter(f, headers)
                    writer.writerow(result)

    df = pd.read_csv('../data_sets/disease_data/disease_data.csv')
    df = df.dropna()
    df = df.sort_values(by=['state', 'year'])
    df.to_csv('../data_sets/disease_data/disease_data.csv', index=None)


def merge_temp_prec_dise_prod_data():
    """

    :return:
    """
    origin_df = pd.read_csv('../data_sets/precipitation_temperature_production.csv')
    disease_df = pd.read_csv('../data_sets/disease_data/disease_data.csv')

    df = pd.merge(origin_df, disease_df, how='inner',
                  on=['state', 'year'])

    df.to_csv('../data_sets/disease_precipitation_temperature_production.csv', index=None)


def main():
    """

    :return:
    """
    # temperature_data_pre_process()
    # precipitation_data_pre_process()
    # merge_precipitation_temperature_production_data()
    # disease_xls_to_csv()
    merge_temp_prec_dise_prod_data()


main()
