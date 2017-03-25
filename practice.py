

a  = [1,2,3,4,5,6,7]

def f1(x):
    a = x
    a[5] = str(x[1])
    return a

a = f1(a)

sum = 0
for num in a:
    if type(num) == type(sum):
        sum += num

print sum