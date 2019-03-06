# Why did I try to analyze account receivables For Finance Factoring
Finance Factoring은 회계상 매출채권(Account Receivables)을 제3자에게 판매하는 행위를 말한다.<br/>
Factoring은 채권을 판매함으로 인하여 금융담보(Security)와 비슷한 성질을 가진다고 생각될 수 있으나, 실제로 금융담보와 성격이 매우 다르다고 할수 있다.<br/>
왜냐하면 Factoring은 즉각적인 유동성 확보 및 회계부채감소의 효과를 기대함에 주목적이 있기 때문이다.<br/><br/>
그래서 제3자는 담보를 위한 계약이 아니기 때문에, Factoring 계약에 앞서 위험을 회피하기 위한 자료를 요청하게 된다. 왜냐하면 질이 나쁜 매출채권을 인수하는 경우, 이자비용 등을 포함한 프리미엄에 문제가 발생하기 때문이다.<br/>
Factoring 계약 전에는 반드시 아래와 같은 자료를 필요로 한다.<br/><br/>

각 개별거래처의 매출채권(Receivables)이 계약대로 이행(입금)이 되었거나, 이행되지 않는 경우 얼마나 지연되었는가를 확인되어야 한다.<br/>
이 절차는 매우 중요하다. 왜냐하면 제3자와 Factoring 계약시, 계약조건 및 수수료 등 비용에 큰 변동이 있을 수 있기 때문이다.(당신이라면 대금결제를 4개월 이상 지연하는 매출채권을 매입하겠는다?)<br/><br/>
코드는 각 기간/ID별 매출의 질(입금이 이행되었는가? 아니면 어느정도 지연되었는가?)을 판단하기 위함이다. (매우 많은 계산을 요구한다)<br/><br/>

* receivables-factoring.py : 분석코드
* receivables-factoring-clustering-visualization.py : 분석자료의 시각화 분석 (K-Means / PCA / t-SNE)

* https://en.wikipedia.org/wiki/Factoring_(finance)
