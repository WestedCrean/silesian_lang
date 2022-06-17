from strings_with_arrows import *
import string

# constants
DIGITS = "0123456789"
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

# errors
class Error:
	def __init__(self, pos_start, pos_end, error_name, details) -> None:
		self.pos_start = pos_start
		self.pos_end = pos_end
		self.error_name = error_name
		self.details = details
	
	def as_string(self):
		res = f"{self.error_name}: {self.details}"
		res += f"\n\Plik {self.pos_start.filename}, linio {self.pos_start.line + 1}"
		res += '\n\n' + string_with_arrows(self.pos_start.filetext, self.pos_start, self.pos_end)
		return res

class IllegalCharError(Error):
	def __init__(self, pos_start, pos_end, details):
		super().__init__(pos_start, pos_end, "Niylegalny znak", details)

class InvalidSyntaxError(Error):
	def __init__(self, pos_start, pos_end, details=''):
		super().__init__(pos_start, pos_end, 'Niynŏleżnŏ  godka', details)

class RTError(Error):
	def __init__(self, pos_start, pos_end, details, context):
		super().__init__(pos_start, pos_end, 'błōnd', details)
		self.context = context

	def as_string(self):
		result  = self.generate_traceback()
		result += f'{self.error_name}: {self.details}'
		result += '\n\n' + string_with_arrows(self.pos_start.filetext, self.pos_start, self.pos_end)
		return result

	def generate_traceback(self):
		result = ''
		pos = self.pos_start
		ctx = self.context

		while ctx:
			result = f'  Plik {pos.filename}, linio {str(pos.line + 1)}, w {ctx.display_name}\n' + result
			pos = ctx.parent_entry_pos
			ctx = ctx.parent

		return 'Śledzynie (Nojnowsze je ôstatnie):\n' + result

# Position

class Position:
	def __init__(self, idx, line, column, filename, filetext):
		self.idx = idx
		self.line = line
		self.column = column
		self.filename = filename
		self.filetext = filetext
	
	def advance(self, current_char=None):
		self.idx += 1
		self.column += 1

		if current_char == '\n':
			self.line += 1
			self.column = 0
		
		return self

	def copy(self):
		return Position(self.idx, self.line, self.column, self.filename, self.filetext)

# Tokens

TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_IDENTIFIER = 'IDENTIFIER'
TT_KEYWORD = 'KEYWORD'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_POW = 'POW'
TT_EQ = 'EQ'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EOF = 'EOF'

KEYWORDS = [
	'VAR'
]

SYMBOL_TO_TOKEN = {
	'+': TT_PLUS,
	'-': TT_MINUS,
	'*': TT_MUL,
	'/': TT_DIV,
	'^': TT_POW,
	'(': TT_LPAREN,
	')': TT_RPAREN,
}

class Token:
	def __init__(self, type_, value=None, pos_start=None, pos_end=None):
		self.type = type_
		self.value = value
		
		if pos_start:
			self.pos_start = pos_start.copy()
			self.pos_end = pos_start.copy()
			self.pos_end.advance()

		if pos_end:
			self.pos_end = pos_end
	
	def matches(self, type_, value):
		return self.type == type_ and self.value == value

	def __repr__(self) -> str:
		if self.value: 
			return f'{self.type}:{self.value}'
		return f'{self.type}'

# Lexer

class Lexer:
	def __init__(self, filename, text):
		self.filename = filename
		self.text = text
		self.pos = Position(-1, 0, -1, filename, text)
		self.current_char = None
		self.advance()
	
	def advance(self):
		self.pos.advance(self.current_char)
		self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None
	
	def make_tokens(self):
		tokens = []
		
		while self.current_char is not None:
			if self.current_char in '\t':
				self.advance()
			elif self.current_char.isspace():
				self.advance()
			elif self.current_char in DIGITS:
				tokens.append(self.make_number())
			elif self.current_char in LETTERS:
				tokens.append(self.make_identifier())
			elif self.current_char in SYMBOL_TO_TOKEN.keys():
				tokens.append(Token(SYMBOL_TO_TOKEN[self.current_char], self.current_char, pos_start=self.pos))
				self.advance()
			else:
				pos_start = self.pos.copy()
				char = self.current_char
				self.advance()
				return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")
		tokens.append(Token(TT_EOF, pos_start=self.pos))
		return tokens, None
	
	def make_number(self):
		number = ''
		dot_count = 0
		pos_start = self.pos.copy()

		while self.current_char is not None and self.current_char in DIGITS + '.':
			if self.current_char == '.':
				if dot_count == 1: break
				dot_count += 1
				num_str += '.'
			else:
				number += self.current_char
			self.advance()
		if dot_count == 0:
			return Token(TT_INT, int(number), pos_start, self.pos)
		else:
			return Token(TT_FLOAT, float(number), pos_start, self.pos)

	def make_identifier(self):
		id_str = ''
		pos_start = self.pos.copy()

		while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
			id_str += self.current_char
			self.advance()

		tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
		return Token(tok_type, id_str, pos_start, self.pos)

# Nodes

class NumberNode:
	def __init__(self, token):
		self.token = token

		self.pos_start = self.token.pos_start
		self.pos_end = self.token.pos_end
		
	def __repr__(self) -> str:
		return f"{self.token}"

class BinOpNode:
	def __init__(self, left_node, op_tok, right_node):
		self.left_node = left_node
		self.op_tok = op_tok
		self.right_node = right_node

		self.pos_start = self.left_node.pos_start
		self.pos_end = self.right_node.pos_end

	def __repr__(self):
		return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
	def __init__(self, op_tok, node):
		self.op_tok = op_tok
		self.node = node

		self.pos_start = self.op_tok.pos_start
		self.pos_end = node.pos_end

	def __repr__(self):
		return f'({self.op_tok}, {self.node})'

# Parse result

class ParseResult:
	def __init__(self):
		self.error = None
		self.node = None
		
	def register(self, res):
		if isinstance(res, ParseResult):
			if res.error:
				self.error = res.error
			return res.node
		return res
		
	def success(self, node):
		self.node = node
		return self
		
	def failure(self, error):
		self.error = error
		return self

# Parser

class Parser:
	def __init__(self, tokens) -> None:
		self.tokens = tokens
		self.tok_idx = -1
		self.advance()
	
	def advance(self):
		self.tok_idx += 1
		if self.tok_idx < len(self.tokens):
			self.current_tok = self.tokens[self.tok_idx]
		return self.current_tok

	def parse(self):
		res = self.expr()
		if not res.error and self.current_tok.type != TT_EOF:
			return res.failure(
				InvalidSyntaxError(
					self.current_tok.pos_start, 
					self.current_tok.pos_end, 
					f"Ôczekowano było '+', '-', '*', '/', '^', '(' lub ')', znaleziōno było: {self.current_tok.value}"))
		return res

	def bin_op(self, func, ops):
		res = ParseResult()
		left = res.register(func())
		if res.error:
			return res

		while self.current_tok.type in ops:
			op = self.current_tok
			res.register(self.advance())
			right = res.register(func())
			left = BinOpNode(left, op, right)

		return res.success(left)

	def factor(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type in [TT_PLUS, TT_MINUS]:
			res.register(self.advance())
			factor = res.register(self.factor())
			if res.error: 
				return res
			return res.success(UnaryOpNode(tok, factor))

		elif tok.type in [TT_INT, TT_FLOAT]:
			res.register(self.advance())
			return res.success(NumberNode(tok))
		
		elif tok.type == TT_LPAREN:
			res.register(self.advance())
			expr = res.register(self.expr())
			if res.error: 
				return res
			if self.current_tok.type == TT_RPAREN:
				res.register(self.advance())
				return res.success(expr)
			else:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Ôczekowano było ')'"
				))

		return res.failure(InvalidSyntaxError(
			tok.pos_start, tok.pos_end,
			"Ôczekowano było liczba cołkŏ lub zmiynno  przecinkowa"
		))


	def term(self):
		return self.bin_op(self.factor, [TT_MUL, TT_DIV])

	def expr(self):
		return self.bin_op(self.term, [TT_PLUS, TT_MINUS])

# runtime result
class RTResult:
	def __init__(self):
		self.value = None
		self.error = None

	def register(self, res):
		if res.error: self.error = res.error
		return res.value

	def success(self, value):
		self.value = value
		return self

	def failure(self, error):
		self.error = error
		return self

# number
class Number:
	def __init__(self, value):
		self.value = value
		self.set_pos()
		self.set_context()

	def set_pos(self, pos_start=None, pos_end=None):
		self.pos_start = pos_start
		self.pos_end = pos_end
		return self
	
	def set_context(self, context=None):
		self.context = context
		return self

	def added_to(self, other):
		if isinstance(other, Number):
			return Number(self.value + other.value).set_context(self.context), None

	def subbed_by(self, other):
		if isinstance(other, Number):
			return Number(self.value - other.value).set_context(self.context), None

	def multed_by(self, other):
		if isinstance(other, Number):
			return Number(self.value * other.value).set_context(self.context), None

	def divided_by(self, other):
		if isinstance(other, Number):
			if other.value == 0:
				return None, RTError(
					other.pos_start, other.pos_end,
					"Dzielynie bez nul",
					self.context
				)
			
			return Number(self.value / other.value).set_context(self.context), None
	
	def __repr__(self):
		return str(self.value)

# context
class Context:
	def __init__(self, display_name, parent=None, parent_entry_pos=None):
		self.display_name = display_name
		self.parent = parent
		self.parent_entry_pos = parent_entry_pos

# interpreter
class Interpreter:
	def visit(self, node, context):
		method_name = f'visit_{type(node).__name__}'
		method = getattr(self, method_name, self.no_visit_method)
		return method(node, context)
	
	def no_visit_method(self, node, context):
		raise Exception(f'No visit_{type(node).__name__} method defined')

	def visit_NumberNode(self, node, context):
		return RTResult().success(
			Number(node.token.value).set_context(context).set_pos(node.pos_start, node.pos_end)
		)
	
	def visit_BinOpNode(self, node, context):
		res = RTResult()

		left = res.register(self.visit(node.left_node, context))
		if res.error: 
			return res

		right = res.register(self.visit(node.right_node, context))
		if res.error: 
			return res

		if node.op_tok.type == TT_PLUS:
			result, error = left.added_to(right)
		elif node.op_tok.type == TT_MINUS:
			result, error = left.subbed_by(right)
		elif node.op_tok.type == TT_MUL:
			result, error = left.multed_by(right)
		elif node.op_tok.type == TT_DIV:
			result, error = left.divided_by(right)

		if error:
			return res.failure(error)
		else:
			return res.success(result.set_pos(node.pos_start, node.pos_end))

	def visit_UnaryOpNode(self, node, context):
		res = RTResult()

		number = res.register(self.visit(node.node, context))
		if res.error: 
			return res

		error = None

		if node.op_tok.type == TT_MINUS:
			number, error = number.multed_by(Number(-1))

		if error:
			return res.failure(error)
		else:
			return res.success(number.set_pos(node.pos_start, node.pos_end))
		

def run(filename, text):
	# generate tokens
	lexer = Lexer(filename, text)
	tokens, error = lexer.make_tokens()
	if error:
		return None, error

	# generate AST
	parser = Parser(tokens)
	ast = parser.parse()

	if ast.error:
		return None, ast.error
	
	# run program
	interpreter = Interpreter()
	context = Context('<program>')
	result = interpreter.visit(ast.node, context)
	return result.value, result.error