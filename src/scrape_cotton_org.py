import requests
import re


def get_state_list(url, headers):
    """

    :param url:
    :param headers:
    :return:
    """
    response = requests.get(url, headers=headers)
    page = response.text
    pattern = re.compile('<option value="([A-Z]+)">\w+</option>', re.S)
    result = re.findall(pattern, page)

    return result


def get_county_list(url, headers, state):
    """

    :param url:
    :param headers:
    :param state:
    :return:
    """
    headers['Content-Length'] = '95'
    headers['Content-Type'] = 'application/x-www-form-urlencoded'
    headers['Origin'] = 'http://www.cotton.org'
    headers['Referer'] = 'http://www.cotton.org/econ/cropinfo/cropdata/county-production-history.cfm'

    form_data = 'STATE_HISTORY=' + state + '&GET_STATE=Choose+State'

    response = requests.post(url, data=form_data, headers=headers)
    page = response.text
    pattern = re.compile('<option value="([A-Z][a-z]+)">\w+</option>', re.S)
    result = re.findall(pattern, page)

    return result


def get_cotton_history_xla(url, headers, state, county):
    """

    :param url:
    :param headers:
    :param state:
    :param county:
    :return:
    """
    headers['Content-Length'] = '95'
    headers['Content-Type'] = 'application/x-www-form-urlencoded'
    headers['Origin'] = 'http://www.cotton.org'
    headers['Referer'] = 'http://www.cotton.org/econ/cropinfo/cropdata/county-production-history.cfm'

    form_data = 'GET_STATE=&STATE_HISTORY=' + state + '&COUNTY=' + county + \
                '&YEAR=1972&TYPE=UPLAND&GET_HISTORY=Show+County+History'

    response = requests.post(url, data=form_data, headers=headers)
    page = response.text
    pattern = re.compile('<a href="(http://www.cotton.org/newams/amsxls/County_Production_History-\d+-\d+-\d+-\d+-\d+.xla)">Export To Excel</a>', re.S)
    result = re.findall(pattern, page)[0]

    file = requests.get(result)

    file_name = state + '-' + county + '.xla'
    open(file_name, 'wb').write(file.content)
    # print(result)


def main():
    """

    :return:
    """
    url = 'http://www.cotton.org/econ/cropinfo/cropdata/county-production-history.cfm'
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'Cookie': '__utmz=22345385.1519164809.1.1.utmccn=(direct)|utmcsr=(direct)|utmcmd=(none); '
                  'CFID=20306594; CFTOKEN=249445060e1ac45e-730EE556-C2B2-5FA9-EF5EBF2D83B7CBF7; '
                  'JSESSIONID=4652C9841C0A10EEAE5058DA71E6FDD9.cfusion; '
                  '__utma=22345385.422969782.1519164809.1519173282.1521470455.3; __utmb=22345385; __utmc=22345385',
        'Host': 'www.cotton.org',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36'
    }

    state_list = get_state_list(url, headers)
    for state in state_list:
        county_list = get_county_list(url, headers, state)
        for county in county_list:
            get_cotton_history_xla(url, headers, state, county)


main()
