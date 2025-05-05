# from openai import OpenAI
# client = OpenAI()
# re = client.completions.create(
#   model="gpt-3.5-turbo-instruct",
#   prompt=["Say this is a test","Say I am the best"],
#   max_tokens=7,
#   temperature=0,
# )

# x1= re.choices[0].text
# y1= re.choices[0].finish_reason
# x2= re.choices[1].text
# print(re)








# import re
# inst = " The following is a\n       list of random words. Select the word that doesn't belong to this list.   "
# inst = re.sub(r"\s+", " ", inst).strip()
# x = re.compile(r'\b{0}\b'.format("following"), flags=re.IGNORECASE).search(inst)
# print(inst)








# result={'prompt': 'Come up with a series of tasks:\n1. Link all the entities in the sentence (highlighted in brackets) to a Wikipedia page. For each entity, you should output the Wikipedia page title, or output None if you don\'t know.\n2. Brainstorm a list of possible New Year\'s resolutions.\n3. Write a sentence that ends with the word "published".\n4. Solving the equation and find the value of X. Show your steps.\n5. Select the oldest person from the list.\n6. As a newly hired sports coach, what are your plans for the first 30 days on the job?\n7. Summarize the following document with several bullet points.\n8. Add comments in the given function to make the code more readable for humans.\n9.', 
#         'response': ' ', 
#         'created_at': '2025-05-02 23:09:05.071054'}
# print(type(result))
# print(type([result]))
# x = [result]*2
# print(x)









# import numpy as np
# rouge_scores = [0.24390243902439024, 0.15384615384615383, 0.20689655172413793, 0.13793103448275862, 0.14814814814814814, 0.07142857142857142, 0.2222222222222222, 0.1935483870967742, 0.24, 0.125, 0.13333333333333333, 0.15384615384615383, 0.0, 0.15789473684210525, 0.14814814814814814, 0.2, 0.1875, 0.08695652173913045, 0.14814814814814814, 0.1904761904761905, 0.06666666666666667, 0.08, 0.2424242424242424, 0.14814814814814814, 0.04444444444444444, 0.08333333333333333, 0.16, 0.0, 0.125, 0.13793103448275862, 0.21428571428571427, 0.15384615384615383, 0.20689655172413793, 0.14285714285714285, 0.0, 0.13333333333333333, 0.15384615384615383, 0.17142857142857143, 0.30769230769230765, 0.1875, 0.11428571428571428, 0.15384615384615383, 0.17142857142857143, 0.13333333333333333, 0.1739130434782609, 0.2, 0.1212121212121212, 0.15384615384615383, 0.1818181818181818, 0.14285714285714285, 0.17647058823529413, 0.1568627450980392, 0.06896551724137931, 0.1851851851851852, 0.23076923076923078, 0.13333333333333333, 0.07692307692307691, 0.2222222222222222, 0.1818181818181818]
# new = np.argsort(rouge_scores)[-10:][::-1]
# print(new)
















# from openai import OpenAI
# client = OpenAI()
# re = client.completions.create(
#   model="gpt-3.5-turbo-instruct",
#   prompt=["Say this is a test"],
#   max_tokens=7,
#   temperature=0,
# )

# x1= dict(re.choices[0])

# print(re)






# from collections import OrderedDict

# # 普通字典
# regular_dict = {}
# regular_dict['a'] = 1
# regular_dict['b'] = 2
# regular_dict['c'] = 3

# # 有序字典
# ordered_dict = OrderedDict()
# ordered_dict['a'] = 1
# ordered_dict['b'] = 2
# ordered_dict['c'] = 3

# # 另一个有序字典，键值对相同但插入顺序不同
# ordered_dict2 = OrderedDict()
# ordered_dict2['c'] = 3
# ordered_dict2['a'] = 1
# ordered_dict2['b'] = 2

# # 输出结果
# print("普通字典:", regular_dict)  # Python 3.7+ 也会按插入顺序
# print("有序字典:", ordered_dict)
# print("有序字典2:", ordered_dict2)

# # 比较结果
# print("普通字典 == 有序字典:", regular_dict == ordered_dict)  # True (键值对相同)
# print("有序字典 == 有序字典2:", ordered_dict == ordered_dict2)  # False (键值对相同但顺序不同)





