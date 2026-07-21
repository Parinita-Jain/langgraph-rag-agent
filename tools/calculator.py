import ast

import operator
from langchain_core.messages import AIMessage
UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

def evaluate(node):

    if isinstance(node, ast.Constant):
        return node.value
    
    if isinstance(node, ast.UnaryOp):

        operand = evaluate(node.operand)

        operator_function = UNARY_OPERATORS.get(type(node.op))

        if operator_function is None:
            raise ValueError("Unsupported unary operator")

        return operator_function(operand)

    if isinstance(node, ast.BinOp):

        left = evaluate(node.left)

        right = evaluate(node.right)

        print("Left:", left)
        print("Right:", right)

        operator_function = OPERATORS.get(type(node.op))
        
        if operator_function is None:
            raise ValueError("Unsupported operator")
        
        return operator_function(left, right)
    raise ValueError("Unsupported expression")


def calculator_node(state):

    print("\n===== CALCULATOR NODE =====")

    expression = state["tool_input"]  

    try:

        tree = ast.parse(expression, mode="eval")

        result = evaluate(tree.body)
       
        return {
                "messages":[
                    AIMessage(content=str(result))
                ],

                "output":{
                    "value":result
                },

                "success":True,

                "error":None
            }

    except Exception as e:

        return {

            "messages":[
                AIMessage(content="Calculator failed.")
            ],

            "output":{},

            "success":False,

            "error":str(e)

        }