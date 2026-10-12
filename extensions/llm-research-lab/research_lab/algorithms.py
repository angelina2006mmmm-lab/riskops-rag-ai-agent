"""Verified classic algorithms and data structures."""
import heapq
def binary_search(seq,val):
 lo,hi=0,len(seq)
 while lo<hi:
  mid=(lo+hi)//2
  if seq[mid]<val:lo=mid+1
  else:hi=mid
 return lo if lo<len(seq) and seq[lo]==val else -1
def two_pointers_pair(seq,target):
 i,j=0,len(seq)-1
 while i<j:
  v=seq[i]+seq[j]
  if v==target:return i,j
  if v<target:i+=1
  else:j-=1
 return None
def top_k_largest(seq,k):return heapq.nlargest(k,seq) if k>0 else []
def balanced_brackets(s):
 stack=[];match={")":"(","]":"[","}":"{"}
 for c in s:
  if c in "([{":stack.append(c)
  elif c in match:
   if not stack or stack.pop()!=match[c]:return False
 return not stack
class Node:
 def __init__(self,key,left=None,right=None):self.key,self.left,self.right=key,left,right
def inorder(root):return [] if root is None else inorder(root.left)+[root.key]+inorder(root.right)
