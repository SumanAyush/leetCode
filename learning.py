from functools import reduce
from typing import List
from collections import Counter
import itertools

class Solution:
    def singleNumber(nums) -> int:
        nums.sort()
        i = 0
        while i < len(nums):
            if len(nums) == 1:
                return nums[0]
            else:
                print(nums[i], nums[i + 1])
                if (nums[i] == nums[i + 1]):
                    i = i + 2
                else:
                    return nums[i]
        return nums[-1]

    def permute(nums):
        result = []

        def backtrack(path, remaining):
            print("PATH :: ", path, "REMAINING :: ", remaining)
            if not remaining:
                result.append(path)
                return

            for i in range(len(remaining)):
                backtrack(path + [remaining[i]], remaining[:i] + remaining[i + 1:])

        backtrack([], nums)
        return result

    def maxProfit(prices) -> int:
        value = 0

        def check(prices):
            nonlocal value
            print(prices)
            if prices == sorted(prices, reverse=True):
                print("SORTED")
                return value
            if prices:
                if prices[-2] > prices[-1]:
                    check(prices[:-2])
                for i in range(len(prices)):
                    print(prices[-1], prices[i], value, " --> ", prices[-1] - prices[i])
                    if prices[-1] - prices[i] > value:
                        value = prices[-1] - prices[i]
                check(prices[:-1])

        check(prices)
        return value

    def maxProfitS(prices) -> int:
        value = 0

        print(" VALUE :: => ")

        if prices:
            for i in range(len(prices)):
                if i < len(prices) - 1:
                    if prices[i] < (max(prices[i + 1:])):
                        if (max(prices[i + 1:]) - prices[i]) > value:
                            value = (max(prices[i + 1:]) - prices[i])
        return value

    def anagram(input):
        return_list = []

        def rec_ana(input):
            dummy_input = list(input)
            if input:
                print(input)
                nonlocal return_list
                internal_list = []
                current_word = sorted(input[0])
                dummy_input.remove(input[0])
                internal_list.append(input[0])
                for i in range(1, len(input)):
                    if sorted(input[i]) == current_word:
                        internal_list.append(input[i])
                        dummy_input.remove(input[i])
                        print("INPUT ", input)
                return_list.append(internal_list)
                rec_ana(dummy_input)

        rec_ana(input)
        return return_list

    def anagram_second(strs):
        lirt = []
        dicA = {}
        for i in range(0, len(strs)):

            if ''.join(sorted(strs[i])) in dicA:
                dicA[''.join(sorted(strs[i]))].append(strs[i])
            else:
                listA = []
                listA.append(strs[i])
                dicA[''.join(sorted(strs[i]))] = listA
        for key in dicA.keys():
            lirt.append(dicA[key])
        return lirt

    def max_sum_subarray(nums) -> int:

        # list_f =[]

        maxSum = 0

        def sub_arr(nums_d, counter):
            counter = counter + 1
            # print(" >> CALLED <<")
            # nonlocal list_f
            nonlocal maxSum
            if nums_d:
                for i in range(1, len(nums_d) + 1):
                    reduced_sum = reduce(lambda x, y=0: x + y, nums_d[0:i])
                    print(nums_d[0:i], " ==> ", reduced_sum)
                    if reduced_sum > maxSum:
                        maxSum = reduced_sum

                    # list_f.append(nums_d[0:i])
            if counter < len(nums):
                sub_arr(nums_d[1:], counter)

        sub_arr(nums, 0)
        return (maxSum)

    def max_sum_subarray_optimized(nums) -> int:
        max_sum = nums[0]
        current_sum = nums[0]

        for i in nums[1:]:
            current_sum = max(i, current_sum + i)
            max_sum = max(max_sum, current_sum)
            print(i, "   ", current_sum, "   ", max_sum)

    def productExceptSelf(nums) -> List[int]:
        output_list = []
        back_mult = 1
        for i in range(1, len(nums) + 1):
            print(" ==> NUMS ", i)
            output_list.append(reduce(lambda x, y: x * y, nums[i:], 1) * back_mult)
            print(" == >  OUTPUT ", output_list)
            back_mult = back_mult * nums[i - 1]
            print(" == > BACK ", back_mult)
        return output_list

    def productExceptSelf_twoPass(nums) -> List[int]:
        n = len(nums)
        output = [1] * n  # Initialize output array with 1s

        # First pass: Compute prefix products
        prefix_product = 1
        for i in range(n):
            output[i] = prefix_product * output[i]
            prefix_product = prefix_product * nums[i]
        print(output)
        #  [1, 1, 2, 6]
        sufix_product = 1
        for i in range(n - 1, -1, -1):
            output[i] = output[i] * sufix_product
            sufix_product = nums[i] * sufix_product
        print(output)
        return output

    def matrixReshape(mat: List[List[int]], r: int, c: int):
        if c * r == (len(mat) * len(mat[0])):

            # res = [[1 for i in range(c)] * r]
            # print (res)
            res = []
            for i in range(0, r):
                res.append([])

            print(res)

            for i in range((len(mat) * len(mat[0]))):
                # For getting the value from original matrix
                row = int(i / len(mat[0]))
                column = int(i % len(mat[0]))
                value = mat[row][column]
                # For inserting it to new list

                row_new = int(i / c)

                column_new = int(i % c)

                print(row_new, column_new)

                res[row_new].append(value)
                print(row_new, column_new, res)

            return res
        else:
            return mat

    def search(nums_s: List[int], target: int) -> int:
        pass

    # def isValid(s: str) -> bool:
    #     is_valid_flag = False
    #     if s == "":
    #         return True
    #     if len(s) % 2 == 0:
    #         def bracketValidator(paren) -> bool:
    #             nonlocal is_valid_flag
    #             print("LENGTH  of PAREN   " , len(paren) , "PAREN :: " , paren)
    #             if paren != "":
    #                 starting_index = 0
    #                 if paren[starting_index] == "(" and ")" in paren:
    #                     ending_index = paren.index(")")
    #                     if "(" in paren[1:ending_index]:
    #                         ending_index = ending_index+paren[ending_index+1:].index(")")
    #                         # print( "ending_index " , ending_index , "   ==>  " , paren[1:ending_index])
    #                 elif paren[starting_index] == "[" and "]" in paren:
    #                     ending_index = paren.index("]")
    #                     if "[" in paren[1:ending_index]:
    #                         print( "ending_index " , ending_index , "   ==>  " , paren[1:ending_index], "  -> ", paren[ending_index+1:] )
    #                         ending_index = ending_index+paren[ending_index+1:].index("]")
    #                 elif paren[starting_index] == "{" and "}" in paren:
    #                     ending_index = paren.index("}")
    #                     if "{" in paren[1:ending_index]:
    #                         ending_index = ending_index+paren[ending_index+1:].index("}")
    #                 else:
    #                     is_valid_flag = False
    #                     return is_valid_flag
    #                 if ending_index != starting_index+1:
    #                     bracketValidator(paren[starting_index+1: ending_index])
    #
    #                 else:
    #                     if len(paren) > 2:
    #                         is_valid_flag = False
    #                         bracketValidator(paren[2:])
    #                     else:
    #                         is_valid_flag = True
    #             return is_valid_flag
    #         bracketValidator(s)
    #     else:
    #         return False
    #     return is_valid_flag

    def isValid(s: str) -> bool:
        s = list(s)
        print("S :: ",s)
        stack_list = []
        closing_paren = {')': '(', ']': '[', '}': '{'}
        for paren in s:
            # print(stack_list , " PAREN " , paren)
            if stack_list and paren in closing_paren:
                if stack_list[-1] == closing_paren.get(paren):
                    stack_list.pop(-1)
                else:
                    return  False
            else:
                stack_list.append(paren)


        return True if not stack_list else False


    def canConstruct(ransomNote: str, magazine: str) -> bool:
        if len(ransomNote) > len(magazine):  # Good optimization
            return False
        magazine_counts = Counter(magazine)  # O(n)
        print(magazine_counts)
        for char in ransomNote:  # O(m)
            if magazine_counts[char] > 0:
                magazine_counts[char] -= 1
            else:
                return False
        return True

    # canConstruct("ab","aab")

    def threeSumNotRecommended( nums: List[int]) -> List[List[int]]:
        finalList = []
        for iters in itertools.combinations(nums,3):
            if iters[0]+iters[1]+iters[2] == 0:
                if sorted(iters) not in finalList:
                    finalList.append(sorted(iters))
        return finalList

    def threeSum( nums: List[int]) -> List[List[int]]:
        result = []
        seen = set()
        nums = sorted(nums)
        print("NUMS :: ",nums)
        for i in range(len(nums)):
            print("++++++++++++++++++++++++++++++++++++++")
            complement_map = {}
            print("complement_map :: ",complement_map)
            for j in range(i + 1, len(nums)):
                complement = -(nums[i] + nums[j])
                print("complement VALUE :: " ,complement)
                if complement in complement_map:
                    triplet = tuple((nums[i], nums[j], complement))
                    if triplet not in seen:
                        result.append(list(triplet))
                        seen.add(triplet)
                else:
                    complement_map[nums[j]] = True
                    print("complement_map IN ELSE ",complement_map)
        return result



    def isBalanced(num: str) -> bool:
        sum_even = 0
        sum_odd = 0
        for i in range(len(num)):
            int_val = int(num[i])
            if i%2 == 0:
                sum_even = sum_even + int_val
            else:
                sum_odd = sum_odd + int_val
        if sum_odd == sum_even:
            return True
        else:
            return False

    # print(isBalanced(""))



    def removeElement(nums: List[int], val: int) -> int:
        i =0
        while len(nums)> i :

            if nums[i] == val:
                nums.remove(val)
                i = i-1
                # print("NUMS : ", nums)
            i = i+1

            # print( len(nums) ,  i )
        return len(nums),nums
    # print(removeElement([0,1,3,0,4,2],2))
    # print(removeElement([1],1))
    # print(removeElement([3,2,2,3],3))
    # print(removeElement([0,1,2,2,3,0,4,2],2))


    def strStr(haystack: str, needle: str) -> int:

        for i in range(len(haystack)-(len(needle))+1):
            print(haystack[i:i+(len(needle))])
            if haystack[i:i+(len(needle))] == needle:
                return i
        return -1


    # print(strStr("a","a"))


    def climbStairs(n: int) -> int:
        if n == 1:
            return 1
        else:
            return (2**(n-2))+1

    # print(climbStairs(6))


    def reverse( x: int) -> int:
        counter = 0
        return_int = ""
        if -2**31 <= x <= 2**31 - 1:
            if x == 0:
                return 0
            elif x < 0:
                x = x*-1
                while x // 10:
                    counter = counter+1
                    return_int = return_int + str(x%10)
                    x = x//10
                return_int = return_int + str(x)

                print("counter:: ",counter)

                if -2**31 <= int(return_int) <= 2**31 - 1:
                    return int(return_int)
                else:
                    return 0
            else:
                while x // 10:
                    counter = counter+1
                    return_int = return_int + str(x % 10)
                    x = x//10
                return_int = return_int + str(x)

                print("counter:: ",counter)
                if -2**31 <= int(return_int) <= 2**31 - 1:
                    return int(return_int)
                else:
                    return 0
        else:
            return 0





    print(reverse(1534236469))




    # print(threeSum([-1,0,1,2,-1,-4]))
    # inputs = ["(([]){})", "" , "([])" , "()" , "()[]{}" , "(]" , "([)]" , "(([])({}))" , "[[[]" ,"[[]]","]" ,"(])"]
    # # inputs = ["(])"]
    # for strs in inputs:
    #     print(strs, "  ::  ", isValid(strs))

    # #input_data = [[9, 2], [3, 4]]
    # input_data = [7,8,1,2,3,4,5,6]
    # # input_data = [-1,1,0,-3,3]
    # # input_data = [1,2,3,4]
    # # return = [["bat"],["nat","tan"],["ate","eat","tea"]]
    # for i in input_data:
    #     print("---------------------------------")
    #     print( i,"  --> ",search(input_data, i))
    # # print( "2  --> ",search(input_data, 1))


