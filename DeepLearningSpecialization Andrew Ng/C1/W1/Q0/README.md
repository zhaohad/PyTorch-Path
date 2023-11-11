# numpy构建基本函数

1. 探索numpy相乘原则
2. 探索numpy点乘原则
3. 探索numpy转置矩阵

# Answer
1. 探索numpy相乘原则
   1. 任意时刻都是同元素位置相乘
   2. 如果维度不匹配就会进行扩展
   3. 扩展原则，对a.shape = (a1, ..., an)、b.shape = (b1, ..., bm)
      1. 做维度匹配，即对n m进行比较，若n m不等，则将小的变量增加维度，增加的新维度元素数量为1
      例如：a.shape = (a1, a2, a3, a4, a5), b.shape = (b1, b2, b3)则将b变为 b.shape = (1, 1, b1, b2, b3)
      2. 经过前一步，维度变为 a.shape = (a1, ... ak), b.shape = (b1, ..., bk), k = max(m, n)
      3. 从最低维度开始匹配，即从从第k维开始比较，匹配成功的条件：ak == bk 或 ak == 1 或 bk == 1
      4. 若ak == bk，则无需扩展。若ak == 1，则将a中k维度元素直接复制bk份，若bk == 1同理
2. 不太好算，用到再说吧，假设 a.shape = (an, ... a2, a1)，b.shape = (bm, ..., b2, b1)。那么numpy.dot(a, b)能算的条件是 a1 == b2。点乘后的维度 numpy.dot(a, b).shape = (an, ..., a2, (没有a1) bm, ..., b3, (没有b2) b1)
   1. 假设a.shape = (A, B, C, D)。b.shape = (E, F, G, H)。numpy.dot(a, b).shape = (A, B, C, E, F, H)