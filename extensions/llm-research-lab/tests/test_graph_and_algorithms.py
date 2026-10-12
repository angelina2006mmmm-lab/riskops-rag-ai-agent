import pytest
from research_lab.graph_rag import load_policies,retrieve,research_answer,evaluate
from research_lab.algorithms import binary_search,two_pointers_pair,top_k_largest,balanced_brackets,Node,inorder
from research_lab.langgraph_adapter import retrieval_node,route
def test_version():
 assert load_policies("2025-10-01")["SUP-01"]["version"]==1
 assert load_policies("2026-10-01")["SUP-01"]["version"]==2
def test_graph_multihop_and_citation():
 hits=retrieve("supplier",top_k=5)
 assert "INC-02" in [h["id"] for h in hits]
 assert next(h for h in hits if h["id"]=="INC-02")["hops"]==3
 assert all(h["path"][-1]=="policy:"+h["id"] for h in hits)
def test_abstain():
 assert research_answer("totally unknown")["status"]=="abstain"
 assert research_answer("supplier","2024-01-01")["status"]=="abstain"
def test_eval():
 r=evaluate()
 assert r["cases"]==10 and r["correct"]==10
def test_search_and_pointers():
 assert binary_search([1,2,2,3],2)==1
 assert two_pointers_pair([1,2,4,6],8)==(1,3)
def test_heap_stack_tree():
 assert top_k_largest([3,8,1,8],2)==[8,8]
 assert balanced_brackets("([]{})") and not balanced_brackets("([)]")
 assert inorder(Node(2,Node(1),Node(3)))==[1,2,3]
def test_langgraph_nodes_without_pkg():
 assert route(retrieval_node({"question":"supplier"}))=="grounded"
def test_real_langgraph():
 pytest.importorskip("langgraph")
 from research_lab.langgraph_adapter import compile_graph
 assert compile_graph().invoke({"question":"supplier"})["status"]=="grounded"
