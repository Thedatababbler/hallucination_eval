# from langchain.llms import LlamaCpp
from langchain.llms import OpenAI, HuggingFaceHub,HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

import os

os.environ["OPENAI_API_KEY"] = "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"


from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4", request_timeout=15)

import pickle
with open('922-Standard-SDv1.pickle', 'rb') as f:
    sdvg = pickle.load(f)

###make test set
prompt_list = list(sdvg.keys())

mark_list = []
count = 0
old_count = 0
for i in range(len(prompt_list)):
    if i % 4 == 0:
        # print(i)
        count += 1
    if count % 10 == 0 or count == 0:
        
        if count == old_count:
            continue
        # print(i)
        old_count = count
        mark_list.append(prompt_list[i].split('-')[1])

test_list = []

for prompt in mark_list[:25]:
    for i in range(4):
        test_list.append(f'{i}_{prompt}')

print(len(test_list))


# from langchain.llms import OpenAI
from langchain.indexes import GraphIndexCreator
from langchain.chains import GraphQAChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.graphs.networkx_graph import KnowledgeTriple

# llm.temperature = 0

# index_creator = GraphIndexCreator(llm=llm)


# for text in texts.split("."):
#   triples = index_creator.from_text(text)
#   print(triples.get_triples())
#   for (node1, relation, node2) in triples.get_triples():
#     final_graph.add_triple(KnowledgeTriple(node1, relation, node2))
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI

# Schema
def get_ent(inp):
    schema = {
        "properties": {
            "object_name": {"type": "string"},
            "shape": {"type": "string"},
            "color": {"type": "string"},
            "texture":{"type":"string"},
            "count":{"type":"integer"},
            "relationship":{"type":"string"},
            "relation subject": {"type":"string"}
        },
        "required": ["object_name", "shape", "color", "texture", "relationship", "count_number"],
    }
    prompt = ChatPromptTemplate(input_variables=['input'], 
    messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], 
        template="Extract and save the relevant entities mentioned in the following passage together with their properties.\n\nOnly extract the properties mentioned in the 'information_extraction' function.\n\nIf a property is not present and is not required in the function parameters, do not include it in the output. For 'object_name' property, do not include any adjectives or expressive word,just keep the noun. i.e.for 'leather wallet' just keep wallet for the object name and save leather for the texture properties.\n\nPassage:\n{input}\n"))])
    # Input
    # inp = """The fluffy pillow was on the left of the striped blanket"""
    # Run chain
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    chain = create_extraction_chain(schema, llm, prompt)
    ents = chain.run(inp)
    return ents

# from collections import defaultdict
# def askQuestions(prompt:str, parsed:list):
#     questions = defaultdict(dict)
#     answers = defaultdict(dict)
#     for d in parsed:
#         name = ''
        
#         for k, v in d.items():
#             if k == 'object_name':
#                 name = v
#             elif k == 'color':
#                 temp = f'What is the {k} of {name}?'
#                 questions[name]['color'] = temp
#                 answers[name]['color'] = v
#             elif k == 'shape':
#                 temp = f'What is the {k} of {name}?'
#                 questions[name]['shape'] = temp
#                 answers[name]['shape'] = v
#             elif k == 'texture':
#                 temp = f'What is the {k} of {name}?'
#                 questions[name]['texture'] = temp
#                 answers[name]['texture'] = v
#             elif k == 'count_number':
#                 count_name = 'count'
#                 temp = f'What is the {count_name} of {name}?'
#                 questions[name]['count'] = temp
#                 answers[name]['count'] = v
#             elif k == 'relationship':
#                 if v not in prompt:
#                     # we only care about the relation mentioned in the original text
#                     # since our KG is directed
#                     continue
#                 subject = d['relation subject']
#                 temp = f'What is the relation between {name} and {subject}?'
#                 questions[name]['relationship'] = temp
#                 answers[name]['relationship'] = v
#     return questions, answers

from collections import defaultdict

def askQuestions(prompt:str, parsed:list, nodelist):
    questions = defaultdict(dict)
    answers = defaultdict(dict)
    # notion = 'Note: If you cannot find any'
    for d in parsed:
        name = ''
        name_list = []
        for k, v in d.items():
            if k == 'object_name':
                name = v
                name_list.append(name)
                for id in range(20):
                    name_id = f'{name}{id}'
                    if name_id in nodelist:
                        name_list.append(name_id)
            elif k == 'color':
                for name in name_list:
                    temp = f'What is the {k} of {name}?'
                    questions[name]['color'] = temp
                    answers[name]['color'] = v
            elif k == 'shape':
                for name in name_list:
                    temp = f'What is the {k} of {name}?'
                    questions[name]['shape'] = temp
                    answers[name]['shape'] = v
            elif k == 'texture':
                for name in name_list:
                    temp = f'What is the {k} of {name}?'
                    questions[name]['texture'] = temp
                    answers[name]['texture'] = v
            # elif k == 'count_number':
            #     count_name = 'count'
            #     temp = f'What is the {count_name} of {name}?'
            #     questions[name]['count'] = temp
            #     answers[name]['count'] = v
            elif k == 'relationship':
                for name in name_list:
                    if v not in prompt:
                        # we only care about the relation mentioned in the original text
                        # since our KG is directed
                        print('!!!!'*10)
                        continue
                    rel_list, rel_subs = [], []
                    subject = d['relation subject']
                    rel_subs.append(subject)
                    for jd in range(20):
                        subject_jd = f'{subject}{jd}'
                        if subject_jd in nodelist:
                            rel_subs.append(subject_jd)
                    
                    for subject in rel_subs:
                        temp = f'What is the relation between {name} and {subject}?'
                        rel_list.append(temp)

                    questions[name]['relationship'] = rel_list
                    answers[name]['relationship'] = v
    return questions, answers

# import pdb; pdb.set_trace()
import tqdm



# qa_dict = defaultdict(dict)
# visited_ents = set()
# for inp in tqdm.tqdm(test_list):

#     original = inp
#     inp = inp.split('_')[1]
    
#     # import pdb; pdb.set_trace()
#     sdvg_key = original.replace('_', '-')
#     g_temp = sdvg[sdvg_key]
#     nodelist = list(g_temp.nodes())
#     if inp not in visited_ents:
#         ents = get_ent(inp)
#         visited_ents.add(inp)
#     qs, answs = askQuestions(inp, ents, nodelist)
#     qa_dict[original]['questions'] = qs
#     qa_dict[original]['answers'] = answs

# with open('qa_dict.pkl', 'wb') as file:
#     pickle.dump(qa_dict, file)

with open('qa_dict.pkl', 'rb') as file:
    qa_dict = pickle.load(file)

###Make KG

# g1 = sdvg['2-The fluffy pillow was on the left of the striped blanket']
def make_triples(sdvg, prompt_name):
    g1 = sdvg[prompt_name]
    g1.nodes()

    triplets = []
    for i, node1 in enumerate(g1.nodes()):
        for j, node2 in enumerate(g1.nodes()):
            #if node1 != node2:
            try:
                # print(node1, node2)
                rel = g1[node1][node2]['relationship']
                # print(rel)
                if '$' in rel:
                    r1, r2 = rel.split('$')
                    r1 = f'is on {r1} of'
                    triplets.append((node1, r1, node2))
                    triplets.append((node1, r2, node2))
                    # for r in rels:
                    #     triplets.append((node1, r, node2))
                else:
                    r = f'has {rel} of' 
                    # if node1 in node2:
                        # node2 = node2.replace(node1+'_', '')
                    triplets.append((node1, r, node2))
            except:
                pass
    return triplets

def get_graph(triplets, f_index_creator):
    final_graph = f_index_creator.from_text('')
    for (node1, relation, node2) in triplets:
        # print(node1, relation, node2)
        final_graph.add_triple(KnowledgeTriple(node1, relation, node2))
    
    return final_graph

# from langchain.prompts import PromptTemplate
# entity_prompt = '''Identify all the concrete nouns in the following sentences. A concrete noun is a noun that represents a physical object that can be perceived by the senses.\
# For instance, \n\nEXAMPLE: 'She heard the sound of jingle of her keys in the deep pocket of her coat,' \nOutput: jingle, keys, pocket, and coat.\nEND OF EXAMPLE\n\nEXAMPLE:\
# 'The dog barked loudly toward the void darkness, which echoed through the empty hallway', \nOutput: dog, hallway\nEND OF EXAMPLE\n\nNow is your turn!\n\n{input}\nOutput:'''
# def marking(ans, gt, question, llm):
#     from langchain.prompts import PromptTemplate
#     template = """Please analyze the following two answers and determine if they express the same meaning, albeit in different words. Provide a direct response with 'Yes' if they mean the same thing, or 'No' if they convey different meanings.\
# \nQuestion: '{question}'\
# \nAnswer_A:'{answer}' \nAnswer_B'{gt}'. 
# Are the two answers the same meaning: Yes or No? """

#     prompt = PromptTemplate(template=template, input_variables=["answer", "gt", "question"])
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
#     res = llm_chain.predict(answer=ans, gt=gt, question=question)
#     return res
def marking(ans, gt, question, llm):
    from langchain.prompts import PromptTemplate
    template = """Please analyze the following two answers and determine if they express the same meaning, albeit in different words. Provide a direct response with 'Yes' if they mean the same thing, or 'No' if they convey different meanings.\
\nQuestion: '{question}'\
\nAnswer_A:'{answer}' \nAnswer_B'{gt}'. 
Are the Answer_A and Answer_B indicate the same answer, though the maybe in different words? Note, if Answer_B is an short-form of Answer_A skipping the subjects and objects in the sentence, you should answer yes. Now, give me your answer: Yes or No? """

    # print(template)
    prompt = PromptTemplate(template=template, input_variables=["answer", "gt", "question"])
    # print(prompt)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    res = llm_chain.predict(answer=ans, gt=gt, question=question)
    if gt in ans and res not in ['Yes', 'yes']:
        return 'Yes'
    return res

def rubric(qtype, mark, total_marks):
    # print('$$$$'*10)
    attributes = ['color', 'shape', 'texture', 'count']
    relation = ['relationship']
    # print(qtype, mark, total_marks)
    if qtype in attributes:
        if mark in ['yes', 'Yes']:
            total_marks = total_marks #make correct prediction no punishiment
        elif mark in ['No', 'no']:
            total_marks -= 1 # attribute only minus one point
    if qtype in relation:
        if mark in ['yes', 'Yes']:
            total_marks = total_marks #make correct prediction no punishiment
        elif mark in ['No', 'no']:
            total_marks -= 4 # attribute only minus one point
    # print(total_marks)
    return total_marks

import copy
###############
####rewrite Questions
def rewrite_qs(g1, qs, answs):
    nodelist = list(g1.nodes())
    new_qs = copy.deepcopy(qs)
    for subj, qdict in qs.items():
        # new_qs[subj] = qdict
        for id in range(10):
            id_subj = f'{subj}{id}'
            
            if id_subj in nodelist:
                # print(id_subj)
                answs[id_subj] = answs[subj]
                for k, v in qdict.items():
                    # new_qs[id_subj] = qdict
                    # print(v, subj, id_subj)
                    new_v = v.replace(subj, id_subj)
                    # print(new_v)
                    # print(new_qs[id_subj])
                    new_qs[id_subj][k] = new_v
                    # print(new_qs[id_subj])
    return new_qs, answs

import re

def remove_digits(s):
    return re.sub(r'\d+', '', s)

from langchain.prompts import PromptTemplate
entity_prompt = '''Identify all the concrete nouns in the following sentences. A concrete noun is a noun that represents a physical object that can be perceived by the senses.\
Note if there is a number after the noun, i.e. chair0 or bed1, just treat them as normal noun and extract them. Do not extract any words related color, shape, and texture.\
For instance, \n\nEXAMPLE: 'pillow1 is on the top of bed0 which is red color.' \nOutput: 'pillow1, bed0'END OF EXAMPLE\n\nEXAMPLE:\
'The black dog0 is walking in frount of a wood bench, and the man0 on the bench is looking at the dog0', \nOutput: 'dog0, bench, man0 '\nEND OF EXAMPLE\n\nNow is your turn!\n\n{input}\nOutput:'''

entity_prompt_temp = PromptTemplate(input_variables=['input'], template=entity_prompt)

# import pdb; pdb.set_trace()
# for inp in test_list:
#     f_index_creator = GraphIndexCreator(llm=llm)
#     inp = inp.replace('_', '-')
#     triplets = make_triples(sdvg, inp)
#     final_graph = get_graph(triplets, f_index_creator)

#     chain = GraphQAChain.from_llm(llm, graph=final_graph, entity_prompt=entity_prompt_temp, verbose=True)
#     # llm.temperature = 0
#     answer = chain.run('pillow0 has color of')




res = defaultdict(dict)

for inp in tqdm.tqdm(test_list):
    f_index_creator = GraphIndexCreator(llm=llm)
    original = inp
    inp = inp.replace('_', '-')
    triplets = make_triples(sdvg, inp)
    final_graph = get_graph(triplets, f_index_creator)
    chain = GraphQAChain.from_llm(llm, graph=final_graph,
                               entity_prompt=entity_prompt_temp, 
                               verbose=False)
    qs, answs = qa_dict[original]['questions'], qa_dict[original]['answers']
    g1 = sdvg[inp]
    new_qs, answs = rewrite_qs(g1, qs, answs)
    # import pdb; pdb.set_trace()
    accuracy = []
    total_marks = 10
    visited = []
    visited_marks = defaultdict(str)
    for subj, qdict in new_qs.items():
        for qtype, query in qdict.items():
            # print(f"\n\tQuestion:{query}")
            ans = chain.run(query)
            # print(f'\n\tOur answers:{ans}')
            gt = answs[subj][qtype]
            # print(f'\n\tGround truth:{gt}')
            mark = marking(ans=ans, gt=gt, question=query, llm=llm)
            # print(f'\n\tMark={mark}')
            # if mark == 'Yes':
            #     accuracy.append(1)
            # else:#
            #     accuracy.append(-1)
            accuracy.append(mark)
            # print('*****'*20)
            if (remove_digits(subj), qtype) in visited:
                # new_marks = rubric(qtype, mark, total_marks)
                # gap = total_marks - new_marks # points lost for a given 
                # if new_marks > visited[remove_digits(subj)]: # new
                #     visited_marks[remove_digits(subj)] = gap
                #     total_marks = new_marks
                print(total_marks)
                if mark in ['Yes', 'yes'] and visited_marks[remove_digits(subj)] not in ['Yes', 'yes']:
                    # print(mark, subj)
                    # print(visited_marks)
                    total_marks = rubric(qtype, mark, total_marks)
                # print(total_marks)

            else:
                total_marks = rubric(qtype, mark, total_marks)
                visited_marks[subj] = mark
                visited.append((subj, qtype))
                # print(total_marks)
            # visited[remove_digits(subj)] =vmark
    res[inp]['acc'] = accuracy
    res[inp]['mark'] = total_marks

with open('res.pickle', 'wb') as f2:
    pickle.dump(res, f2)