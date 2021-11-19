## lyrics analysis / EDA
- process : embedding -> auto encoder (hidden state) -> 2 dim data -> clustering
- objective : korean ballad analysis for topic extraction
- first step : embedding, cbow

<h3>
Insight
</h3>
<ol>
    <li>
        한 문장에 대한 워드임베딩 벡터의 평균들은 해당 노래의 성격을 잘 타나낼 수 있음.    
    </li>
    <li>
        Doc2Vec 원리를 이해하게 되었음.
    </li>
    <li>
        Doc2Vec이 LSTM이 발전하게 된 하나의 아이디어가 아니었나 싶다. 평균화 작업을 거치지만 더하기 연산, 곱하기 연산이 있는 LSTM의 벡터연산과 비슷함.
    </li>
    <li>
        벡터에 대한 경험을 할 수 있었음.
    </li>
</ol>