import pytest
from research_lab.graph_rag import research_answer,load_policies,retrieve
from research_lab.algorithms import binary_search,two_pointers_pair,balanced_brackets,inorder,Node
from research_lab.neural_baselines import dataset
def test_empty_graph_query():
 assert research_answer("")["status"]=="abstain"
def test_future_policy_not_loaded():
 assert load_policies("2024-01-01")=={}
def test_invalid_date():
 with pytest.raises(ValueError):retrieve("supplier","invalid")
def test_old_version():
 r=retrieve("supplier","2025-10-01")
 assert next(x for x in r if x["id"]=="SUP-01")["version"]==1
def test_search_miss():
 assert binary_search([],1)==-1 and binary_search([2,4,6],3)==-1
def test_pointer_and_tree_edge():
 assert two_pointers_pair([1],2) is None and inorder(None)==[]
def test_bracket_and_data_reproducible():
 assert not balanced_brackets("([)]")
 a,_=dataset(30,123);b,_=dataset(30,123)
 assert a.equal(b)
