```python
import numpy as np, scipy as sp, random as rd

def cross_entropy(p, q):
    """Require that p and q are numpy arrays"""
    result = -p.dot(np.log(q))
    return result
```


```python
def random_discrete_pdf(length, res=int(1e6)):
    """Compute a random multinomial 
    PDF with `length` entries"""
    bs = sorted(rd.sample(list(range(1, res)), length - 1))
    pdf = []
    pdf.append(bs[0])
    pairs = zip(bs, bs[1:])
    
    for a, b in pairs:
        pdf.append(b - a)
        
    pdf.append(res-b)
    pdf = np.array(pdf)
    pdf = pdf/pdf.sum()
    return pdf
```


```python
length = 6
res = int(1e6)
p = random_discrete_pdf(length, res)
q = random_discrete_pdf(length, res)
u = np.array([1.0]*length)
u = u/u.sum()
cxpq = cross_entropy(p, q)
cxpp = cross_entropy(p, p)
cxqq = cross_entropy(q, q)
cxuu = cross_entropy(u, u)
cxuq = cross_entropy(u, q)
cxup = cross_entropy(u, p)
print('p')
print(p)
print(p.sum())
print()
print('q')
print(q)
print(q.sum())
print()
print('cx(p, q)')
print(cxpq)
print()
print('cx(p, p)')
print(cxpp)
print()
print('cx(q, q)')
print(cxqq)
print()
print('cx(u, u)')
print(cxuu)
print()
print('cx(u, p)')
print(cxup)
print()
print('cx(u, q)')
print(cxuq)
```

    p
    [0.225033 0.334608 0.023639 0.242121 0.144237 0.030362]
    1.0
    
    q
    [0.205184 0.17357  0.246404 0.109577 0.007623 0.257642]
    1.0000000000000002
    
    cx(p, q)
    2.255409102412797
    
    cx(p, p)
    1.5192827855972078
    
    cx(q, q)
    1.602963091358229
    
    cx(u, u)
    1.791759469228055
    
    cx(u, p)
    2.1967233033644336
    
    cx(u, q)
    2.196617096486305



```python
-np.log(0.01)
```




    4.605170185988091




```python

```


```python

```


```python

```


```python

```
