####################################################################################
# Factoring은 회계상 매출채권(Account Receivables)을 제3자에게 판매하여, 즉각적인 유동성 확보 및 회계부채감소의 효과를 기대한다.
# 제3자는 Factoring 계약에 앞서, 과거 및 현재 채권의 거래안전성을 확인할 필요가 있다.
# 제3자는 각 개별거래처의 매출채권(Receivables)이 계약대로 이행(입금)이 되었거나, 이행되지 않는 경우 얼마나 지연되었는가를 확인하는 것이다.
# 이 절차는 매우 중요하다. 왜냐하면 제3자와 Factoring 계약시, 계약조건 및 수수료 등 비용에 큰 변동이 있을 수 있기 때문이다.(당신이라면 대금결제를 4개월 이상 지연하는 매출채권을 매입하겠는다?)
# 아래 코드는 각 기간/ID별 매출의 질(입금이 정상적으로 이행되었는가? 아니면 어느정도 지연되었는가?)을 판단하기 위함이다.
# * https://en.wikipedia.org/wiki/Factoring_(finance)
#
#
# # 분석에 대한 기본설명
# 만약 ABC라는 거래처에서 2017.01에 100만원 매출이 발생했다. ABC는 1달 매출액을 다음달 말일에 입금하는 계약을 체결했다.
# 매출 
#  2017.01 : 100만원
# 입금
#  2017.02 : 20만원 (정상회입)
#  2017.03 : 10만원 (1개월 지연 회입)
#  2017.04 : 50만원 (2개월 지연 회입)
#  2017.05 : 20만원 (3개월 지연 회입)
# 
# 100만원 매출 중, 정상적으로 입금된 금액은 20만원 밖에 안된다. 심지어 최종입금완료는 3개월이 지연되었다.
# 아래 코드는 ABC 같은 자료를 분석하기 위해서이다.
####################################################################################

import pandas as pd
import numpy as np
import calendar
from pandas import DataFrame, Series
from datetime import datetime, date, time, timedelta

### 자료읽기
# UTF-8 으로 인코딩된 CSV 자료
dataRaw = pd.read_csv('./csv/date.csv')
dataRaw['date'] = pd.to_datetime(dataRaw['date']) # date를 날짜형으로 변경

dataRaw.info()
### 변수설명
# id : 고객코드
# date : 날짜(월말기준)
# base : 기초잔고
# rev : 매출
# dep : 입금
# end : 기말잔고
# dueDateMonth : 매출발생 이후, 입금일 (단위는 1달 기준) // ie: 2.0 - 2달 뒤 입금

### 매월 말 날짜만 추출
# 직관적으로 이해하기는 어렵겠지만, Factoring은 ID별 월매출액의 입금을 기간별로 분해야한다.
dates = dataRaw.date.unique()
print(dates) # dates는 2016-01 부터 2018-07 까지 월별 말일 날짜가 저장되었다.

### 1달 단위로 id를 추출 후, 해당 id 매출(revenue)의 회입기간을 분석
# 예전에 작성한 코드로 함수로 쪼개야 함에도 불구하고, 귀찮아서 그냥 작성된 코드를 정리만 했음.
# 매출채권에 대한 이해가 낮은 경우, 상당히 코드 해석이 어려울 수 있음

def dataSlicingByDate(dataset0, dateEnd):
    dataset = dataset0.copy()
    # 1년간 채권자료를 추출
    date1 = dateEnd.copy()
    date2 = date1 + np.timedelta64(350, 'D')
    data = dataset.query("@date1 <= date < @date2").copy()

    # 메모리 관리를 위하여, 기준날짜(date1)에 존재하는 ID만 추출
    singularID = data.loc[data['date'] == date1, 'id'].unique().tolist()
    data = data.loc[data['id'].isin(singularID)].copy()

    data.sort_values(['id', 'date'], inplace=True) # ID와 날짜를 기준으로 정렬
    return data


def dataTreatDateBaseDep(dataset0):
    dataset = dataset0.copy()
    ## 매출이 발생한 날짜를 기준으로 index를 추출
    index = dataset.groupby(['id'])['date'].apply(lambda x: x.idxmin())
    
    ## dateRev에 매출이 발생한 날자를 일괄설정
    dataset['dateRev'] = dataset.loc[index, 'date']
    dataset['dateRev'] = dataset['dateRev'].fillna(method='ffill')

    ## 기초잔고(base)가 음수(-)인 경우 -> 입금액(dep)에 기초잔고(base)를 반영하고, 기초잔고(base)를 0으로 만든다
    dataset.loc[index, 'dep'] = dataset.loc[index, 'dep'].where(dataset.loc[index, 'base'] >= 0, dataset.loc[index, 'dep'] - dataset.loc[index, 'base'])
    dataset.loc[index, 'base'] = dataset.loc[index, 'base'].where(dataset.loc[index, 'base'] >= 0, 0)

    ## 상기 반영된 기초잔고(base)를 baseModified라는 열에 반영한다
    dataset['baseModified'] = dataset.loc[index, 'base']
    dataset['baseModified'] = dataset['baseModified'].fillna(method='ffill')

    ## ID별 입금액(dep)의 누적합계(depCumulative)를 구한 후, 차분(depCumulativeDiff)한다
    dataset['depCumulative'] = dataset.groupby(['id'])['dep'].transform(lambda x: np.cumsum(x))
    dataset['depCumulativeDiff'] = dataset.groupby(['id'])['depCumulative'].apply(lambda x: x.diff().fillna(0))

    ## 반영된 기초잔고(baseModified)를 입금에 반영하기 위한 중간단계
    ## (기준월에 이전에 발생된 매출을 입금에서 제외하면, 기준월 매출발생의 입금월을 확인할 수 있기 때문)
    # 입금에 기초잔고액 입금을 반영하기 위한 중간단계로 depAdd 열을 생성
    dataset['depAdd'] = dataset['dep'].where(dataset['baseModified'] >= dataset['depCumulative'], dataset['baseModified'] - (dataset['depCumulative'] - dataset['depCumulativeDiff'])) 
    
    ## depAdd가 음수(-)로 발생된 인덱스를 찾은 후, 매출기준월(dateRev)와 날짜(date)가 같은 경우
    ## depAdd에 기초잔고(base)을 반영한다
    index1 = dataset.loc[dataset['depAdd'] < 0].index
    dataset.loc[index1, 'depAdd'] = dataset.loc[index1, 'depAdd'].where(dataset.loc[index1, 'dateRev'] != dataset.loc[index1, 'date'], dataset.loc[index1, 'base'])
    
    ## 최종적으로 기초잔고액의 입금을 다 반영한 입금액을 계산한다
    dataset['depAdd'] = dataset['depAdd'].where(dataset['depAdd'] >= 0, 0)
    dataset['depModified'] = dataset['dep'] - dataset['depAdd']
    dataset['dep'] = dataset['depModified']
    
    ## 계산필요없는 열을 제외한다
    dataset.drop(['base', 'end', 'baseModified', 'depCumulative', 'depCumulativeDiff', 'depAdd', 'depModified'], axis=1, inplace=True)
    return dataset


def dataTreatRev(dataset0):
    dataset = dataset0.copy()
    ## 매출(rev)이 음수(-)인 인덱스를 찾고(반품에 해당된다), 해당 금액을 입금(dep)으로 반영한다
    ## 그리고 매출(rev)은 0으로 변경한다
    index = dataset[dataset['rev'] < 0].index
    dataset.loc[index, 'dep'] = dataset.loc[index, 'dep'] - dataset.loc[index, 'rev']
    dataset.loc[index, 'rev'] = 0

    ## 입금(dep)이 음수(-)인 인덱스를 찾고, 매출(rev)에 반영한다. 그리고 입금(dep)은 0으로 변경한다
    index = dataset[dataset['dep'] < 0].index
    dataset.loc[index, 'rev'] = dataset.loc[index, 'rev'] - dataset.loc[index, 'dep']
    dataset.loc[index, 'dep'] = 0

    ## 기준일자(dateRev)의 매출(rev)을 revTemp에 저장한다
    index1 = dataset.groupby(['id'])['date'].apply(lambda x: x.idxmin())
    dataset['revTemp'] = dataset.loc[index1, 'rev']
    dataset['revTemp'] = dataset['revTemp'].fillna(method='ffill')

    ## 입금(dep)에 대한 누적합(depCumulative) 및 누적합의 차분(depCumulativeDiff)를 구한다)
    dataset['depCumulative'] = dataset.groupby(['id'])['dep'].apply(lambda x: np.cumsum(x))
    dataset['depCumulativeDiff'] = dataset.groupby(['id'])['depCumulative'].apply(lambda x: x.diff()).fillna(0)

    ## 기준일자 매출액을 입금 누적합과 비교하여, 매출액의 입금을 월별로 확인한다.
    dataset['depModified'] = dataset['dep'].where(dataset['revTemp'] >= dataset['depCumulative'], dataset['revTemp'] - (dataset['depCumulative'] - dataset['depCumulativeDiff']))

    index = dataset.loc[dataset['depModified'] < 0].index
    dataset.loc[index, 'depModified'] = dataset.loc[index, 'depModified'].where(dataset.loc[index, 'dateRev'] != dataset.loc[index, 'date'], dataset.loc[index, 'rev'])
    dataset['depModified'] = dataset['depModified'].where(dataset['depModified'] >=0, 0)

    dataset['dep'] = dataset['depModified']
    dataset['rev'] = dataset['revTemp']

    dataset.drop(['revTemp', 'depModified', 'depCumulative', 'depCumulativeDiff'], axis=1, inplace=True) # 필요없는 열을 삭제
    return dataset


def dataOverDue(dataset0):
    dataset = dataset0.copy()
    ## dueDateMonth : 매출발생 이후, 입금일 (단위는 1달 기준)을 입금된 날짜와 비교하여, 각 입금월의 연체기간을 계산한다
    for dueDate in dataset['dueDateMonth'].unique().tolist():
        i = dataset.loc[dataset['dueDateMonth'] == dueDate].index
        j = dataset.loc[i, 'dateRev'] + timedelta(days=30) * dueDate - timedelta(days=10)
        j = j.apply(lambda x: date(x.year, x.month, calendar.monthrange(x.year, x.month)[1]))

        dataset.loc[i, 'overDue'] = (dataset.loc[i, 'date'] - pd.to_datetime(j)).apply(lambda x: x.days)
    
    ## 연체일자(overDue / 사실상 1달 단위날짜가 나옴)가 음수(-)인 경우 0으로 변경
    dataset['overDue'] = dataset['overDue'].where(dataset['overDue'] >= 0, 0)

    ## 연체일자(overDue)를 기준으로 A-H로 분류된다. (A: 정상회입 / B: 1달이내 연체 / C: 2달이내 연체 ...)
    dataset['overDueLabel'] = pd.cut(dataset['overDue'], bins=[0, 25, 55, 85, 115, 145, 175, 195, np.inf], labels=list('ABCDEFGH'), include_lowest=True).astype('str')
    return dataset


def dataAgg(dataset0):
    dataset = dataset0.copy()
    ##@@@ ID와 날짜(date)를 기준으로 Pivot테이블을 구하고, 자료와 Join한다
    datasetPivot1 = dataset.pivot_table(index=['id', 'date'], columns='overDueLabel', values='dep', aggfunc='sum').fillna(0).reset_index()
    datasetPivot2 = dataset[dataset['date'] == dataset['dateRev']].drop(['dateRev', 'dep', 'overDue', 'overDueLabel'], axis=1)
    datasetPivot2 = datasetPivot2.merge(datasetPivot1, on='id', how='left')
    return datasetPivot2


def dataMain1(dataset0):
    dataset = dataset0.copy()
    dataReturn = pd.DataFrame()
    dates = dataset['date'].unique()

    for i in dates:
        temp = dataSlicingByDate(dataset, i)
        temp = dataTreatDateBaseDep(temp)
        temp = dataTreatRev(temp)
        temp = dataOverDue(temp)
        temp = dataAgg(temp)
        dataReturn = dataReturn.append(temp)
    
    return dataReturn.fillna(0)


def dataMain2(dataset0):
    dataset = dataset0.copy()
    datasetAgg = dataset.groupby(['id', 'date_x']).agg({'rev':'max', 'A':'sum', 'B':'sum', 'C':'sum', 'D':'sum', 'E':'sum', 'F':'sum', 'G':'sum', 'H':'sum'}).reset_index() # id와 날짜를 기준으로 Groupby
    datasetAgg['I'] = datasetAgg['rev'] - datasetAgg[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']].sum(axis=1) # 회입되지 못한 금액을 계산
    datasetAgg = datasetAgg[datasetAgg['rev'] != 0] # 매출이 발생하지 않은 기간(월)은 제외
    return datasetAgg


####################################################
dataFactoring1 = dataMain1(dataRaw)
dataFactoring1.info()
dataFactoring2 = dataMain2(dataFactoring1)
dataFactoring2.info()
dataFactoring2.to_csv('./output/data-factoring.csv', index=False)
# id : ID
# date_x : 매출발생월
# rev : 발생매출액
# A : 원금회수(정상)
# B : 원금회수(1개월)
# C : 원금회수(2개월)
# D : 원금회수(3개월)
# E : 원금회수(4개월)
# F : 원금회수(5개월)
# G : 원금회수(6개월)
# H : 원금회수(6개월초과)
# I : 미회수
####################################################