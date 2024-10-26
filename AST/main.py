from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient,errors
import json
import logging
from fastapi.middleware.cors import CORSMiddleware


# Configure logging
logging.basicConfig(level=logging.DEBUG)
# Logger setup for printing messages
logger = logging.getLogger(__name__)
app = FastAPI()
# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend address
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)
# MongoDB Setup using the provided connection string
# client = MongoClient("mongodb+srv://Lava:Lava7259@cluster0.tg53wqj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0/")
# db = client.rule_engine_db
# rules_collection = db.rules
# rule_counter_collection = db.rule_counter
try:
    client = MongoClient("mongodb+srv://Lava:Lava7259@cluster0.tg53wqj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0/")
    db = client.rule_engine_db
    rules_collection = db.rules
    rule_counter_collection = db.rule_counter
    logger.info("Successfully connected to MongoDB")
except errors.ConnectionFailure as e:
    logger.error(f"Could not connect to MongoDB: {e}")
    raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")

# Initialize the counter if not already present
if rule_counter_collection.count_documents({}) == 0:
    rule_counter_collection.insert_one({"counter": 0})

# Define the AST Node class
class Node:
    def __init__(self, type, value, left=None, right=None):
        self.type = type
        self.value = value
        self.left = left
        self.right = right

    def to_dict(self):
        return {
            'type': self.type,
            'value': self.value,
            'left': self.left.to_dict() if self.left else None,
            'right': self.right.to_dict() if self.right else None
        }

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return None
        return cls(
            type=data['type'],
            value=data['value'],
            left=cls.from_dict(data['left']),
            right=cls.from_dict(data['right'])
        )

# Helper function to parse rule string into an AST
def parse_rule_string(rule_string):
    tokens = rule_string.replace('(', ' ( ').replace(')', ' ) ').split()

    def parse_expression():
        stack = [[]]
        for token in tokens:
            if token == '(':
                stack.append([])
            elif token == ')':
                expr = stack.pop()
                stack[-1].append(expr)
            elif token in ['AND', 'OR']:
                stack[-1].append(token)
            else:
                stack[-1].append(token)

        def build_tree(expr):
            if isinstance(expr, list):
                if len(expr) == 1:
                    return build_tree(expr[0])
                elif 'OR' in expr:
                    idx = expr.index('OR')
                    return Node('operator', 'OR', build_tree(expr[:idx]), build_tree(expr[idx + 1:]))
                elif 'AND' in expr:
                    idx = expr.index('AND')
                    return Node('operator', 'AND', build_tree(expr[:idx]), build_tree(expr[idx + 1:]))
            return Node('operand', ' '.join(expr))

        return build_tree(stack[0])

    return parse_expression()

# Helper function to evaluate the AST against the data
def evaluate_ast(ast, data):
    if ast.type == 'operator':
        if ast.value == 'AND':
            return evaluate_ast(ast.left, data) and evaluate_ast(ast.right, data)
        elif ast.value == 'OR':
            return evaluate_ast(ast.left, data) or evaluate_ast(ast.right, data)
    
    elif ast.type == 'operand':
        try:
            left, op, right = ast.value.split()
            left_value = data.get(left)

            # Check if the left value exists in the data
            if left_value is None:
                logging.error(f"Missing value for '{left}' in the data.")
                return False  # Return False if data is missing the required field
            
            right_value = int(right) if right.isdigit() else right.strip("'")
            
            # Perform comparison based on the operator
            if op == '>':
                return left_value > right_value
            elif op == '<':
                return left_value < right_value
            elif op == '=':
                return left_value == right_value
        except Exception as e:
            logging.error(f"Error evaluating operand: {e}")
            return False  # Return False if an error occurs during evaluation (invalid data)
    
    return False  # Default return False if AST type is unrecognized or there's an issue

# Pydantic model for requests
class RuleCreateRequest(BaseModel):
    rule_string: str

class RuleEvaluateRequest(BaseModel):
    rule_id: int
    data: dict

class RuleCombineRequest(BaseModel):
    rule_ids: list

# Generate a unique, human-readable rule_id
def generate_rule_id():
    counter_doc = rule_counter_collection.find_one_and_update(
        {}, {"$inc": {"counter": 1}}, return_document=True
    )
    return counter_doc['counter']

# API to create a rule
@app.post("/create_rule/")
async def create_rule(request: RuleCreateRequest):
    rule_string = request.rule_string
    ast = parse_rule_string(rule_string)
    rule_id = generate_rule_id()

    rule_data = {
        "rule_id": rule_id,
        "rule_string": rule_string,
        "ast": ast.to_dict()
    }

    rules_collection.insert_one(rule_data)
    
    # Reconstruct AST as a string
    def ast_to_string(ast):
        if ast['type'] == 'operator':
            return f"({ast_to_string(ast['left'])} {ast['value']} {ast_to_string(ast['right'])})"
        elif ast['type'] == 'operand':
            return ast['value']
    
    ast_string = ast_to_string(ast.to_dict())
    
    return {
        'id': rule_id,
        'ast': ast_string
    }

# API to combine multiple rules using AND
@app.post("/combine_rules/")
async def combine_rules(request: RuleCombineRequest):
    rule_ids = request.rule_ids
    rules = rules_collection.find({"rule_id": {"$in": rule_ids}})
    
    asts = [Node.from_dict(rule['ast']) for rule in rules]
    
    combined_ast = asts[0]
    for ast in asts[1:]:
        combined_ast = Node('operator', 'AND', combined_ast, ast)
    
    combined_rule_id = generate_rule_id()
    combined_rule_string = " AND ".join([rule["rule_string"] for rule in rules])

    combined_rule_data = {
        "rule_id": combined_rule_id,
        "rule_string": combined_rule_string,
        "ast": combined_ast.to_dict()
    }

    rules_collection.insert_one(combined_rule_data)

    def ast_to_string(ast):
        if ast['type'] == 'operator':
            return f"({ast_to_string(ast['left'])} {ast['value']} {ast_to_string(ast['right'])})"
        elif ast['type'] == 'operand':
            return ast['value']

    combined_ast_string = ast_to_string(combined_ast.to_dict())

    return {
        'id': combined_rule_id,
        'combined_ast': combined_ast_string
    }

# API to evaluate a rule against the provided data
@app.post("/evaluate_rule/")
async def evaluate_rule(request: RuleEvaluateRequest):
    try:
        # Find the rule by rule_id
        rule = rules_collection.find_one({"rule_id": request.rule_id})
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        # Parse the rule's AST from the database
        ast = Node.from_dict(rule["ast"])
        
        # Evaluate the rule against the provided data
        result = evaluate_ast(ast, request.data)
        
        # Return the result of evaluation (True or False)
        return {"result": result}
    
    except Exception as e:
        logging.error(f"Error evaluating rule: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
