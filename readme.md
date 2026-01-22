hey fam is oli

install streamlit to use the dashboard 
    pip install streamlit pandas 
    OR 
    pip3 install streamlit pandas 
    
Make sure you run the program first

run the dashboard with 
     python -m streamlit run dashboard.py  
     OR 
     python3 -m streamlit run dashboard.py
     OR 
     streamlit run dashboard.py

Whichever works for you

#1 Hard greedy
python3 main_sa.py --helpers helpers_rare.csv --tickets tickets_hard.csv --scheduler greedy
#2 Hard greedy+sa
python3 main_sa.py --helpers helpers_rare.csv --tickets tickets_hard.csv --scheduler greedy+sa
#3 Easy greedy
python3 main_sa.py --helpers helpers_easy.csv --tickets tickets_small.csv --scheduler greedy
#4 Easy greedy+sa
python3 main_sa.py --helpers helpers_easy.csv --tickets tickets_small.csv --scheduler greedy+sa
#5 Random Hard
python3 main_sa.py --helpers helpers_rare.csv --tickets tickets_hard.csv --scheduler random
#6 Random Easy
python3 main_sa.py --helpers helpers_easy.csv --tickets tickets_small.csv --scheduler random
