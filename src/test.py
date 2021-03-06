import re
import csv

headers = ['state', 'county', 'station', 'year',
           'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
           'plant_acres', 'harvest_acres', 'yield_acres', 'bales']

with open('./temperature_production.csv', 'w') as f:
    writer = csv.DictWriter(f, headers)
    writer.writeheader()

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

county = 'Bibb'
state = 'AL'
station = 'USC00010160'

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




