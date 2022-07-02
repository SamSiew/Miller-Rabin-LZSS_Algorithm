"""
Name: Ming Shern, Siew
studentid: 28098552
"""

import sys
from bitarray import bitarray

"""
-------------------------------------------------------------
Huffman
-------------------------------------------------------------
"""
class MinHeap:

    def __init__(self):
        """
        :param count: number of item to be inserted to this heap
        Time Complexity:
            Best Case:  O(k)
            Worst Case: O(k)
            where k is kth value specify by user.
            Best Case == Worst Case, this function will always create a list for N item .
        Space Complexity:
            Best Case:  O(k)
            Worst Case: O(k)
            where N is Maximum number of item that can be inserted to this class.
            Best Case == Worst Case, this function will always create a list for N item which require 4 bytes each N items.
        """
        self.count = 0
        self.array = []

    def __len__(self):
        """
        :return: self.count
        Time Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
            Best Case == Worst Case, this function is returning only number of item in This Heap excluding the first element,
            so that computation can be convienience.
        Space Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
            Best Case == Worst Case, this function will always return a byte for any number.
        """
        return self.count

    def hasItemLeft(self):
        """
        :return: boolean value, true if number of vertex in graph is not empty
                                false if number of vertex in graph is empty
        Time Complexity: O(1), O(1) time is needed to find if length of this heap is zero
        Space Complexity: O(1), O(1) space is needed to find if length of this heap is zero and
                                return boolean value
        """
        return len(self) > 0

    def heapify(self, arr):
        for i in range((len(arr) // 2) - 1, -1, -1):
            self.heapify_aux(arr, i)
        self.array = arr
        self.count = len(arr)

    def heapify_aux(self, arr, i):
        l = 2 * i + 1
        r = 2 * i + 2

        n = len(arr)

        min_child = i

        if l < n and arr[l] != None and arr[l] < arr[min_child]:
            min_child = l

        if r < n and arr[r] != None and arr[r] < arr[min_child]:
            min_child = r

        if min_child != i:
            arr[i], arr[min_child] = arr[min_child], arr[i]  # swap
            self.heapify_aux(arr, min_child)

    def swap(self, i, j):
        """
        :param i: index of an item in array
        :param j: index of an item in array
        :return: no return, it is void method.
        Time Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
            Best Case == Worst Case, this function is swapping position of two item in array.
        Space Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
            Best Case == Worst Case, no memory required for this operation since it does not create any memory for swapping.
        """
        self.array[i], self.array[j] = self.array[j], self.array[i]

    def rise(self, k):
        """
        :param k: index of an item, mainly which is index of new item added to heap.
        :return: nothing to return, it is just swapping method for new item to fit in this heap.
        Time Complexity:
            Best Case:  O(1)
            Worst Case: O(log k)
            where k is number of non-empty item in list.

            Best Case: this function can terminate early if new child nodes is already bigger than parent nodes
            and if new child nodes is equal to parent nodes but already have bigger index value that it holds.
            Therefore, no swapping is required at all.
            Worst Case: when new child nodes is the smallest item of the heaps or the new child nodes is equal to all parent nodes
            but have a bigger index. Therfore, any time the new child nodes encounter any above cases, swapping is reuired at log n times
        Space Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
            Best Case == Worst Case, no memory required for this operation since it does not create any memory for swapping.
        """
        # rise item to parent nodes if parents is already bigger than current new nodes
        while k > 0 and self.array[self.getparent(k)] > self.array[k]:
            self.swap(k, self.getparent(k))
            k = self.getparent(k)
        # swapping parent with new nodes when it equals but have higher index value.
        while k > 0 and self.array[self.getparent(k)] == self.array[k] and self.array[self.getparent(k)] > self.array[k]:
            self.swap(k, self.getparent(k))
            k = self.getparent(k)

    def add(self, item):
        """
        :param wordFreq: a value defines word count to be inserted in to array
        :param counter: index of that item in original list
        :return: nothing, just a void method
        Time Complexity:
            Best Case:  O(1)
            Worst Case: O(log k)
            where k is number of non-empty item in list.
            Best Case: this function can terminate early if new child nodes is already bigger than parent nodes
                and if new child nodes is equal to parent nodes but already have bigger index value that it holds.
                Therefore, no swapping is required at all.
            Worst Case: when new child nodes is the smallest item of the heaps or the new child nodes is equal to all parent nodes
                but have a bigger index. Therfore, any time the new child nodes encounter any above cases, swapping is reuired at log n times
        Space Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
            Best Case == Worst Case, no memory required for this operation since it does not create any memory for swapping.
        """
        # add item when current class length is less than heap array length
        self.array.append(item)
        self.count += 1
        self.rise(self.count - 1)

    def getmin(self):
        """
        :return: get minimum value of frequency of an array which always located at index 1.
        Time Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
        Best Case == Worst Case, access item of array which only tooks O(1).
        Space Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
        Best Case == Worst Case, access item of array which only tooks O(1). Therefore, a byte memory is all its needs.
        """
        return self.array[0]

    def extract(self):
        """
        :return:
        Time Complexity:
            Best Case:  O(log k)
            Worst Case: O(log k)
            where k is number of non-empty item in list.
            BestCase == Worst Case: Happens when new roots node is needed to swap all the way to other item to become a leaf node.
        Space Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
        Best Case == Worst Case, This operation is just swapping operation(in-place function). Therefore, no memory is required to
        perform it.
        """
        # store item to new variable, swap last child nodes with root then reduce length of array
        item = self.array[0]
        self.swap(0, self.count - 1)
        self.array.pop()
        self.count -= 1
        # sink method is called to adjust array to heap structure.
        if self.hasItemLeft():
            self.sink(0)
        return item

    def sink(self, k):
        """
        :param k: index of an item in array
        :return: nothing, just a void method
        Time Complexity:
            Best Case:  O(log k)
            Worst Case: O(log k)
            where k is number of non-empty item in list.
            When last item is swapped with root, it always require comparing with child node and needs to sink all the way to become leaf node
            because in heap structure, child node in scallability is always bigger than any parent nodes of older root node
        Space Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
        Best Case == Worst Case, This operation is just swapping operation(in-place function) therefore, no memory is required to
        perform it.
        """
        # if array has left item, loops continute
        while self.getleft(k) < self.count:
            child = self.getsmallestchild(k)
            # when parent nodes is already smaller than child node, break
            if self.array[k] < self.array[child]:
                break
            # when parent nodes is already equal to child node, then index value with higher is place at root
            elif self.array[k] == self.array[child]:
                break
            self.swap(child, k)
            k = child

    def getparent(self, k):
        """
        :param k: refers to index of item in array
        :return: computation of parent of k index in array.
        Time Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
        Best Case == Worst Case, this function just performing arithmetic operation.
        Space Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
        Best Case == Worst Case, for this function just performing arithmetic operation. Therefore, a byte memory is all its needs
        """
        return k // 2

    def getleft(self, k):
        """
        :param k: refers to index of item in array
        :return: computation to get left child nodes
        Time Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
        Best Case == Worst Case, this function just performing arithmetic operation.
        Space Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
        Best Case == Worst Case, for this function just performing arithmetic operation. Therefore, a byte memory is all its needs
        """
        return 2 * k + 1

    def getright(self, k):
        """
        :param k: refers to index of item in array
        :return: computation to get right child nodes
        Time Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
        Best Case == Worst Case, this function just performing arithmetic operation.
        Space Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
        Best Case == Worst Case, for this function just performing arithmetic operation. Therefore, a byte memory is all its needs
        """
        return 2 * k + 2

    def getsmallestchild(self, k):
        """
        :param k: index of item in array
        :return: either left index of k item or right item of k index
        Time Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
            Best Case == Worst Case, this function just comparing which child nodes is smaller when last nodes is swap with minimum
            after extract() method is called.
        Space Complexity:
            Best Case:  O(1)
            Worst Case: O(1)
            Best Case == Worst Case, This function only involve storing index into variable and accessing item in array.
            Therefore, two byte memory is all its needs resulting O(1)
        """
        left = self.getleft(k)
        right = self.getright(k)
        # check if there is length of class is equal to root with left item only or if left child is smaller, return left if true
        if left == len(self) - 1 or self.array[left] < self.array[right]:
            return left
        # check if there left child node is equal to right node and index value of left is bigger, return left if true
        elif self.array[left] == self.array[right] and self.array[left] < self.array[right]:
            return left
        # return right when we knew right node is assume smaller.
        else:
            return right

    def sort(self):
        arr = []
        while self.hasItemLeft():
            arr.append(self.extract().data)
        return arr


class BinaryNode:
    def __init__(self, data, length, char=None, left=None, right=None):
        self.data = data
        self.length = length
        self.char = char
        self.left = left
        self.right = right

    def isleaf(self):
        return self.left == None and self.right == None

    def __lt__(self, other):
        return other == None or (self.data < other.data) or (self.data == other.data and self.length <= other.length)

    def __gt__(self, other):
        return other == None or (self.data > other.data) or (self.data == other.data and self.length >= other.length)

    def __eq__(self, other):
        return other != None and self.data == other.data and self.length == other.data

    def __len__(self):
        return self.length

    def __iadd__(self, other):
        self.data += other
        return self

    def __str__(self):
        return str(self.char) + ',' + str(self.data) + ',' + str(self.length) + '\n left:' + str(
            self.left) + '\n right:' + str(self.right)


class Node:
    def __init__(self, data, next=None, length=1):
        self.data = data
        self.next = next
        self.count = length

    def __len__(self):
        return self.count

    def __str__(self):
        return str(self.data)


class EncodingNode:
    def __init__(self):
        self.head = None
        self.tail = self.head

    def add(self, node):
        current = node.tail
        current.next = self.head
        self.head = node.head


class LinkedEncoded:
    def __init__(self):
        self.dictionary = [None] * 128
        self.count = 0

    def append(self, char, data):
        if self.dictionary[char] == None:
            self.dictionary[char] = bitarray()
            # self.dictionary[char] = []
            self.dictionary[char].append(data)
            self.count += 1
        else:
            self.dictionary[char].append(data)

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        return self.dictionary[item]

    def represent(self):
        for i in range(128):
            if self.dictionary[i] != None:
                print(chr(i), self.dictionary[i])

def get_code(node, stack, dictionary):
    if node.isleaf():
        symbol = node.char
        for i in range(len(stack)):
            dictionary.append(symbol, stack[i])
    else:
        stack.append(0)
        get_code(node.left, stack, dictionary)
        stack.pop()

        stack.append(1)
        get_code(node.right, stack, dictionary)
        stack.pop()


"""
-------------------------------------------------------------
Elias
-------------------------------------------------------------
"""


class binary:
    def __init__(self, decimal=-1):
        self.decimal = decimal
        self.binary = -1
        self.count = 0
        self.set_to_bin()

    def __len__(self):
        return self.count

    def set_to_bin(self):
        binary_num = 0
        decimal = self.decimal
        i = 0
        while (decimal != 0):
            binary_num += (decimal % 2) * (10 ** i)
            decimal = decimal // 2
            i += 1
        self.count = i
        self.binary = binary_num
        return binary_num

    def merge(self, binary):
        length = len(binary)
        self.count += length
        self.binary = (10 * length * self.binary) + binary.binary


class LinkedBinary:
    def __init__(self, decimal=-1):
        self.decimal = decimal
        self.head = None
        self.tail = None
        self.count = 0
        self.set_to_bin()

    def __len__(self):
        return self.count

    def set_to_bin(self):
        self.head = Node(self.decimal % 2)
        self.tail = self.head
        decimal = self.decimal // 2
        i = 1
        while (decimal != 0):
            newnode = Node(decimal % 2, self.head, i + 1)
            self.head = newnode
            decimal = decimal // 2
            i += 1
        self.count = i

    def encode(self):
        length = self.count - 1
        while length != 0:
            new_bin = LinkedBinary(length)
            new_bin.head.data = 0
            self.merge(new_bin)
            length = len(new_bin) - 1

    def merge(self, binary):
        current = binary.tail
        current.next = self.head
        self.head = binary.head
        self.count += len(binary)


"""
LZSS-Matching
"""


class LempelZiv:
    def __init__(self, pat):
        self.pat = pat
        self.z_array = [0 for i in range(len(pat[0]))]

    def __getitem__(self, item):
        return self.z_array[item]

    def perform(self, start_pos, buffer_size, window_size):
        """
        :param pattern: string representing pat
        :return: array with length of pat

        Time Complexity:
           Worst Case: O(m)
           where m is length of pat
           Z Array is computed m times because unnecessary comparison are removed using z-interval box.
        Space complexity:
           Worst Case: O(m)
           where m is length of pat
           Z Array with m space at most is created in algorithm.
        """
        self.z_array = [0 for i in range(len(self.pat[0]))]
        n = buffer_size
        left, right = -1, -1
        currentposition = start_pos + 1
        if start_pos + n > len(self.pat[0]): n = len(self.pat[0]) - start_pos
        self.z_array[start_pos] = n
        self.perform_z_algorithm(start_pos, currentposition, n, left, right)
        currentposition = start_pos - window_size
        if currentposition < 0: currentposition = 0
        return self.perform_z_algorithm(start_pos, currentposition, n, left, right)

    def perform_z_algorithm(self, start_pos, currentposition, n, left, right):
        maximum_z = currentposition
        while currentposition < start_pos + n:
            # explicit check since there is no box for memorization
            if currentposition > right:
                left, right = currentposition, currentposition
                while right < start_pos + n and right - left < n and self.pat[0][start_pos + right - left] == self.pat[0][right]:
                    right += 1
                zvalue = right - left
            else:
                # explicit check starting from right side since zarray[k:m] != zarray[k-l:m]:
                if self.z_array[start_pos + currentposition - left] == right - currentposition + 1:
                    left, right = currentposition, right + 1
                    while right < start_pos + n and right - left < n and self.pat[0][start_pos + right - left] == self.pat[0][right]:
                        right += 1
                    zvalue = right - currentposition
                else:
                    # smartly copy value since it is within the box or already over compute for k+1 iteration
                    if self.z_array[start_pos + currentposition - left] < right - currentposition + 1:
                        self.z_array[currentposition] = self.z_array[start_pos + currentposition - left]
                    else:
                        self.z_array[currentposition] = right - currentposition + 1
                    if self.z_array[currentposition] != 0 and currentposition < start_pos and self.z_array[currentposition] >= self.z_array[maximum_z]: maximum_z = currentposition
                    currentposition += 1
                    continue
            # only overwrite value when zarray has enough memorization.
            if zvalue > 0:
                self.z_array[currentposition] = zvalue
                right = right - 1
                left = currentposition
                if currentposition < start_pos and zvalue >= self.z_array[maximum_z]:maximum_z = currentposition
            else:
                self.z_array[currentposition] = 0
            currentposition += 1
        return maximum_z


def perform_algorithm(filename, W, L):
    # count frequency

    file = open(filename, 'r')

    pat = [file.read()]

    file.close()

    dicts = [None for i in range(128)]

    for i in range(len(pat[0])):
        char = ord(pat[0][i])
        if dicts[char] == None:
            dicts[char] = BinaryNode(1, 1, char)
        else:
            dicts[char] += 1

    # clean dicts
    i = 127
    while dicts[i] == None:
        dicts.pop()
        i -= 1

    for i in range(len(dicts) - 1, -1, -1):
        if dicts[i] == None:
            dicts[i], dicts[-1] = dicts[-1], dicts[i]
            dicts.pop()

    # build min heap or heapify.
    header = MinHeap()
    header.heapify(dicts)

    newitem = None

    encoding = LinkedEncoded()
    # huffman algo
    while header.hasItemLeft():
        item1 = header.extract()
        item2 = header.extract()

        newitem = BinaryNode(item1.data + item2.data, item1.length + item2.length, left=item1, right=item2)

        if header.hasItemLeft():
            header.add(newitem)

    storage = bitarray()

    get_code(newitem, storage, encoding)

    lz = LempelZiv(pat)

    bit1 = bitarray(1)
    bit1[0] = 1

    bit0 = bitarray(1)
    bit0[0] = 0

    memo = [[bit1, encoding[ord(pat[0][0])]]]

    i = 1

    while i < len(pat[0]):
        pos = lz.perform(i,L,W)

        if lz[pos] == 0:
            memo.append([bit1, encoding[ord(pat[0][i])]])
            i += 1
            continue

        if lz[pos] >= 3:
            memo.append([bit0, get_elias(i - pos, bitarray()), get_elias(lz[pos],bitarray())])
        else:
            memo.append([bit1, encoding[ord(pat[0][i + lz[pos] - 1])]])
        i += lz[pos]

    with open('output_encoder_lzss.bin', 'wb+') as file:
        out = bitarray()
        # header
        bitval = [pow(2, i) for i in range(7)]
        out = get_elias(len(encoding), out)
        for i in range(128):
            if encoding[i] != None:
                charbit = i
                for j in range(6, -1, -1):
                    if charbit >= bitval[j]:
                        charbit -= bitval[j]
                        out.append(1)
                    else:
                        out.append(0)
                out = get_elias(len(encoding[i]), out)
                for j in range(len(encoding[i])):
                    out.append(encoding[i][j] and 1)

        # encode data
        out = get_elias(len(memo), out)

        for i in range(len(memo)):
            for j in range(len(memo[i])):
                out.append(memo[i][j] and 1)
        out.fill()
        out.tofile(file)
        file.close()

def get_elias(num, bitarrays):
    bin = LinkedBinary(num)
    bin.encode()
    current = bin.head
    while current != None:
        bitarrays.append(current.data and 1)
        current = current.next
    return bitarrays

if __name__ == "__main__":
    filename = sys.argv[1]
    w = sys.argv[2]
    l = sys.argv[3]
    perform_algorithm(filename, w, l)


