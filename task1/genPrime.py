"""
Name: Ming Shern, Siew
studentid: 28098552
"""
import sys
import random
import math

class ModuloExponentiate:
    def __init__(self, decimal , numofbit, modulo):
        self.decimal = decimal
        self.numofbit = numofbit
        self.modulo = modulo

    def make_memo(self):
        memo = [0 for i in range(self.numofbit + 1)]
        memo[0] = self.decimal% self.modulo
        for i in range(1, self.numofbit + 1):
            memo[i] = (memo[i-1] * memo[i-1]) % self.modulo
        return memo

def is_even(n):
    if n % 2 == 0:
        return True
    return False

def get_bin_length(n):
    bin_len = 0
    while n != 0:
        bin_len += 1
        n = n // 2
    return bin_len

def millerRabinRandomizedPrimality(n, k):
    """
    :param n: is the number being tested for primality
    :param k: a parameter that determines accuracy of the test
    :return: (True: prime or False: composite)
    """

    if is_even(n): return False

    s = 0
    t = n-1
    t_pos = []

    while (is_even(t)):
        s = s + 1
        if (t % 2 == 1):
            t_pos.append(s)
        t = t //2

    t_len = get_bin_length(t) - 1

    for j in range(k):
         a = random.randint(2, n-1)

         moduloDictionary = ModuloExponentiate(a, s, n).make_memo()

         if (pow(moduloDictionary[s], t) % n != 1): return False

         for i in range(s):
            if ( pow(moduloDictionary[i], t) % n == 1 and pow(moduloDictionary[i-1], t) % n != (n-1 or 1)):
                return False

    return True

def approx_prime_count(n):
    return n/math.log(n)

def genPrime(k):
    N = None
    retval = False
    numofloop = approx_prime_count(2 ** (k) - 1) - approx_prime_count(2 ** (k - 1))
    numofloop = int(numofloop)

    while retval == False:
        N = random.randrange(2 ** (k - 1) + 1,2 ** (k) - 1,2)
        retval = millerRabinRandomizedPrimality(N, numofloop + 1)

    print(N)

if __name__=="__main__":
    k = sys.argv[1]
    genPrime(int(k))
