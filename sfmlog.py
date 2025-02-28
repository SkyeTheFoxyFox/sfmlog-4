import sys, argparse, pathlib, re, pymsch, math, random

def _error(text: str, token):
    print(f"ERROR at ({token.line},{token.column}): {text}")
    sys.exit(2)

class SFMlog:
    def __init__(self):
        pass

    def transpile(self, code: str, cwd: pathlib.Path) -> str:
        tokenizer = _tokenizer(code)
        parser = _parser(tokenizer.tokens, cwd)
        schem_builder = _schem_builder()
        executer = _executer(parser.code, {}, "_", schem_builder)
        executer.as_root_level()
        executer.execute()
        schem_builder.make_schem()
        return schem_builder.schem

class _tokenizer:
    SUB_INSTRUCTION_MAP = {
        "draw": [True],
        "control": [True],
        "radar": [True, True, True, True],
        "op": [True],
        "lookup": [True],
        "jump": [False, True],
        "ucontrol": [True],
        "uradar": [True, True, True, True],
        "ulocate": [True, True],
        "getblock": [True],
        "setblock": [True],
        "status": [True, True],
        "setrule": [True],
        "message": [True],
        "cutscene": [True],
        "effect": [True],
        "fetch": [True],
        "setmarker": [True],
        "makemarker": [True],
        "pop": [True],
        "spop": [True],
        "if": [True],
        "while": [True]
    }

    LINK_BLOCKS = ["gate", "foundation", "wall", "container", "afflict", "heater", "conveyor", "duct", "press", "tower", "pad", "projector", "swarmer", "factory", "drill", "router", "door", "illuminator", "processor", "sorter", "spectre", "parallax", "cell", "electrolyzer", "display", "chamber", "mixer", "conduit", "distributor", "crucible", "message", "unloader", "refabricator", "switch", "bore", "bank", "accelerator", "disperse", "vault", "point", "nucleus", "panel", "node", "condenser", "smelter", "pump", "generator", "tank", "reactor", "cultivator", "malign", "synthesizer", "deconstructor", "meltdown", "centrifuge", "radar", "driver", "void", "junction", "diffuse", "pulverizer", "salvo", "bridge", "acropolis", "dome", "reconstructor", "separator", "citadel", "concentrator", "mender", "lancer", "source", "loader", "duo", "melter", "crusher", "fabricator", "redirector", "disassembler", "gigantic", "incinerator", "scorch", "battery", "tsunami", "arc", "compressor", "assembler", "smite", "module", "bastion", "segment", "constructor", "ripple", "furnace", "wave", "foreshadow", "link", "mine", "scathe", "canvas", "diode", "extractor", "fuse", "kiln", "sublimate", "scatter", "cyclone", "titan", "turret", "lustre", "thruster", "shard", "weaver", "huge", "breach", "hail"]

    class token:
        def __init__(self, type: str, value, line: int = 0, column: int = 0, scope = None):
            self.type: str = type
            self.value = value
            self.line = line
            self.column = column
            self.scope = scope

        def __repr__(self):
            if self.type in ["identifier", "label"]:
                return f'{self.type}({self.scope}{self.value})[{self.line},{self.column}]'
            else:
                return f'{self.type}({self.value if self.value != '\n' else r'\n'})[{self.line},{self.column}]'

        def __str__(self):
            if self.type in ["identifier", "label"]:
                return str(self.scope) + str(self.value)
            elif self.type == "global_identifier":
                return f"global_{str(self.value)}"
            else:
                return str(self.value)

        def with_scope(self, scope: str):
            if self.scope == None:
                return _tokenizer.token(self.type, self.value, self.line, self.column, scope=scope)
            else:
                return self

        def at_pos(self, pos: tuple[int, int]):
            return _tokenizer.token(self.type, self.value, pos[0], pos[1], scope=self.scope)

        def resolve_string(self):
            if self.type == "string":
                return self.value[1:-2]
            else:
                return self.value

    def __init__(self, code: str):
        self.tokens: list[token] = self.tokenize(code)

    def tokenize(self, code: str) -> list[token]:
        tokens = []
        regex = r"(?:(?: |\t)+)|(?:\n+$)|(\n)\n*|(?:#.*)(?:\n|$)|(\".+?\")(?:\s*?$| )|(.+?)(?:\s*?$| )"
        prev_instruction = ""
        prev_token_type = "line_break"
        dist_from_prev_instruction = 0
        for match in re.finditer(regex, code, flags=re.M):
            if not all(x is None for x in match.groups()):
                match_string = [x for x in match.groups() if x != None][0]

                line = 0
                column = 0
                for index, value in enumerate(code):
                    column += 1
                    if index >= match.start():
                        break
                    if value == "\n":
                        line += 1
                        column = 0
    
                token_type, token_value = self.identify_token(match_string, prev_token_type, prev_instruction, dist_from_prev_instruction, (line + 1, column))
                dist_from_prev_instruction += 1
                if token_type == "instruction":
                    prev_instruction = match_string
                    dist_from_prev_instruction = 0
                if not (token_type == "line_break" and prev_token_type == "line_break"):
                    tokens.append(self.token(token_type, token_value, line + 1, column))
                prev_token_type = token_type
        tokens.append(self.token("line_break", "\n", line + 1, column))
        return tokens

    def identify_token(self, string: str, prev_token_type: str, prev_instruction: str, dist_from_prev_instruction: int, pos: tuple[int, int]) -> tuple[str, str | float]:
        if(string == '\n'):
            return ("line_break", "\n")

        if(string[0] == '"' and string[-1] == '"'):
            return ("string_literal", string)

        if(string[0] == '"' or string[-1] == '"'):
            print(f"ERROR at ({pos[0]},{pos[1]}): String not closed")
            sys.exit(2)

        if(string[0] == '%'):
            return ("color_literal", string)

        if(re.search(r"^0x[0-9a-fA-F]*$", string)):
            return ("number", float(int(string[2:], 16)))

        if(re.search(r"^0b[01]*$", string)):
            return ("number", float(int(string[2:], 2)))

        if(re.search(r"^-?[0-9]*(\.[0-9]*)?$", string)):
            return ("number", float(string))

        if string == "true":
            return ("number", 1.0)

        if string == "false":
            return ("number", 0.0)

        if(re.search(r"^-?[0-9]*(\.[0-9]*)?e-?[0-9]*(\.[0-9]*)?$", string)):
            return ("number", float(string))

        if(string[0] == '@'):
            return ("content_literal", string)

        if((string.rstrip("1234567890") in self.LINK_BLOCKS) and (string != string.rstrip("1234567890"))):
            return ("link_literal", string)

        if(prev_token_type == "line_break"):
            if(string[-1] == ':'):
                return ("label", string)
            else:
                return ("instruction", string)

        if(prev_instruction in self.SUB_INSTRUCTION_MAP):
            if(dist_from_prev_instruction < len(self.SUB_INSTRUCTION_MAP[prev_instruction]) and self.SUB_INSTRUCTION_MAP[prev_instruction][dist_from_prev_instruction]):
                return ("sub_instruction", string)

        if(string[0] == '$'):
            return ("global_identifier", string[1:])

        if(string  == "null"):
            return ("defined_literal", string)

        return ("identifier", string)

    def token_list_to_str(tokens: list[token]) -> str:
        string = ""
        last_token = _tokenizer.token("line_break", "\n", 0, 0)
        for token in tokens:
            if token.type == "line_break" or last_token.type == "line_break":
                string += str(token)
            else:
                string += " " + str(token)
            last_token = token
        return string

class _parser:

    def __init__(self, code: list[_tokenizer.token], cwd: pathlib.Path):
        self.cwd = cwd
        self.code = code
        self.imports = []

        self.parse()

    def parse(self):
        self.get_imports()

    def get_imports(self, in_code=None, cwd=None):
        if in_code is None:
            in_code = self.code
            self.code = []
        if cwd is None:
            cwd = self.cwd
        code_iter = iter(in_code)
        line = self.read_line(code_iter)
        while line != []:
            if self.list_get_token(line, 0).value == "import":
                import_path = pathlib.Path(self.list_get_token(line, 1).resolve_string())
                if not import_path.is_absolute():
                    import_path = cwd / import_path
                import_path = import_path.resolve()
                if import_path not in self.imports:
                    self.imports.append(import_path)
                    try:
                        with open(import_path, "r") as f:
                            import_code = f.read()
                    except FileNotFoundError:
                        _error("File not found", self.list_get_token(line, 1))
                    tokenizer = _tokenizer(import_code)
                    self.get_imports(tokenizer.tokens, import_path.parent)
            else:
                self.code.extend(line)

            line = self.read_line(code_iter)

    def read_line(self, code_iter) -> list[_tokenizer.token]:
        line = []
        try:
            token = next(code_iter)
        except StopIteration:
            return []
        while token.type != "line_break":
            line.append(token)
            token = next(code_iter)
        line.append(token)
        return line

    def list_get_token(self, token_list: list[_tokenizer.token], index: int):
        try:
            token = token_list[index]
        except IndexError:
            _error("Unexpected end of line", token_list[-1])
        if token.type == "line_break":
            _error("Unexpected end of line", token)
        return token

class _executer:
    PROC_INSTRUCTIONS = ["proc", "repproc"]
    CONDITIONS = ["equal", "notEqual", "lessThan", "greaterThan", "lessThanEq", "greaterThanEq", "strictEqual"]

    class InstructionLine:
        def __init__(self, tokens):
            self.tokens = tokens

        def require(self, index: int):
            try:
                out = self.tokens[index]
            except IndexError:
                _error(f"Instruction {self.tokens[0].value} expected argument at position {index}", self.tokens[-1])
            if out.type == "line_break":
                _error(f"Instruction {self.tokens[0].value} expected argument at position {index}", out)
            return out

        def option(self, index, default = None):
            if default is None:
                default = _tokenizer.token("defined_literal", "null", self.tokens[-1].line, self.tokens[-1].column)
            try:
                return self.tokens[index]
            except IndexError:
                return default

        def has(self, index):
            try:
                return self.tokens[index].type != "line_break"
            except IndexError:
                return False

        def __getitem__(self, index):
            return self.require(index)

        def __contains__(self, index):
            return self.has(index)

        def __len__(self):
            return len(self.tokens)

    class Macro:
        def __init__(self, name, code, args):
            self.name: str = name
            self.code: list[_tokenizer.token] = code
            self.args: list[_tokenizer.token] = args

    def __init__(self, code: list[_tokenizer.token], global_vars: dict[str, _tokenizer.token], scope_str: str, schem_builder):
        self.code: list[_tokenizer.token] = code
        self.instructions = self.read_lines()
        self.output: list[_tokenizer.token] = []
        self.scope_str = scope_str
        self.macros: dict[str, Macro] = {}
        self.macro_run_counts: dict[str, int] = {}
        self.vars: dict[str, _tokenizer.token] = {}
        self.global_vars: dict[str, _tokenizer.token] = global_vars
        self.allow_mlog = True
        self.is_root = False
        self.schem_builder = schem_builder

        self.exec_pointer = 0

    def execute(self):
        while True:
            if self.exec_pointer >= len(self.instructions):
                break
            inst = self.instructions[self.exec_pointer]

            match inst[0].value:
                case "block":
                    var_name = inst[1]
                    if var_name.type not in ["identifier", "global_identifier"]:
                        _error("Invalid variable name", var_name)
                    block_type = inst[2]
                    if block_type.type != "content_literal":
                        _error("Expected block type", block_type)
                    block_pos = None
                    block_rot = 0
                    if 4 in inst:
                        if self.resolve_var(inst[3]).type != "number":
                            _error("Expected numeric value", inst[3])
                        if self.resolve_var(inst[4]).type != "number":
                            _error("Expected numeric value", inst[4])
                        block_pos = (int(self.resolve_var(inst[3]).value), int(self.resolve_var(inst[4]).value))
                    if 5 in inst:
                        if type(self.resolve_var(inst[5]).value) != float:
                            _error("Expected numeric value", inst[5])
                        block_rot = int(self.resolve_var(inst[5]).value)

                    block = self.schem_builder.Block(inst, block_type, block_pos, block_rot)
                    link_name = self.schem_builder.add_block(block)
                    self.write_var(var_name, _tokenizer.token("block_var", link_name, 0, 0))
                case "proc":
                    proc_code = self.read_till("endproc", self.PROC_INSTRUCTIONS)
                    if proc_code is None:
                        _error("'endproc' expected, but not found", inst[0])
                    proc_executer = _executer(proc_code, self.global_vars, "_", self.schem_builder)
                    proc_executer.macros = self.macros
                    proc_executer.execute()
                    if 3 in inst:
                        pos = (inst[2], inst[3])
                    else:
                        pos = None
                    proc_name = self.schem_builder.add_proc(self.schem_builder.Proc(_tokenizer.token_list_to_str(proc_executer.output), pos))
                    if 1 in inst:
                        self.write_var(inst[1], _tokenizer.token("block_var", proc_name, 0, 0))
                case "repproc":
                    pass
                case "defmac":
                    mac_code = self.read_till("endmac", ["defmac"])
                    if mac_code is None:
                        _error("'endmac' expected, but not found", inst[0])
                    if inst[1].type != "identifier":
                        _error("Invalid name for macro", inst[1])
                    mac_args = []
                    for arg in inst.tokens[2:-1]:
                        if arg.type != "identifier":
                            _error("Invalid name for macro argument", arg)
                        mac_args.append(arg)
                    self.macros[inst[1].value] = self.Macro(inst[1].value, mac_code, mac_args)
                case "mac":
                    if inst[1].value in self.macros:
                        mac = self.macros[inst[1].value]
                        if mac.name not in self.macro_run_counts:
                            self.macro_run_counts[mac.name] = 0
                        mac_executer = _executer(mac.code, self.global_vars, f"{self.scope_str}{mac.name}_{self.macro_run_counts[mac.name]}_", self.schem_builder)
                        for index, arg in enumerate(mac.args):
                            var_token = inst[index + 2]
                            mac_executer.write_var(arg, self.resolve_var(var_token))
                        mac_executer.macros = self.macros.copy()

                        self.macro_run_counts[mac.name] += 1
                        mac_executer.execute()
                        self.output.extend(mac_executer.output)
                    else:
                        _error(f"Unknown macro '{inst[1].value}'", inst[1])
                case "pset":
                    self.write_var(inst[1], self.resolve_var(inst[2]))
                case "pop":
                    self.write_var(inst[2], self.eval_math(inst[1], self.resolve_var(inst[3]), self.resolve_var(inst[4])))
                case "if":
                    code_sections = self.read_sections("endif", ["if"], ["elif", "else"])
                    if code_sections is None:
                        _error("'endif' expected, but not found", inst[0])

                    for instruction, code_block in code_sections:
                        if instruction[0].value == "else" or self.eval_condition(instruction[1], self.resolve_var(instruction[2]), self.resolve_var(instruction.option(3))).value:
                            block_executer = _executer(code_block, self.global_vars, self.scope_str, self.schem_builder)
                            block_executer.macros = self.macros
                            block_executer.vars = self.vars
                            block_executer.allow_mlog = self.allow_mlog
                            block_executer.macro_run_counts = self.macro_run_counts
                            block_executer.execute()
                            self.output.extend(block_executer.output)
                            break
                case "while":
                    code_block = self.read_till("endwhile", ["while"])
                    if code_block is None:
                        _error("'endwhile' expected, but not found", inst[0])
                    while self.eval_condition(inst[1], self.resolve_var(inst[2]), self.resolve_var(inst.option(3))).value:
                        block_executer = _executer(code_block, self.global_vars, self.scope_str, self.schem_builder)
                        block_executer.macros = self.macros
                        block_executer.vars = self.vars
                        block_executer.allow_mlog = self.allow_mlog
                        block_executer.macro_run_counts = self.macro_run_counts
                        block_executer.execute()
                        self.output.extend(block_executer.output)
                case "for":
                    code_block = self.read_till("endfor", ["for"])
                    if code_block is None:
                        _error("'endfor' expected, but not found", inst[0])
                    if inst[1].value == "range":
                        if 5 in inst:
                            if int(self.coerce_num(self.resolve_var(inst[5]))) == 0:
                                _error("'for range' step value must not be zero", inst[5])
                            iter_range = range(int(self.coerce_num(self.resolve_var(inst[3]))), int(self.coerce_num(self.resolve_var(inst[4]))), int(self.coerce_num(self.resolve_var(inst[5]))))
                        elif 4 in inst:
                            iter_range = range(int(self.coerce_num(self.resolve_var(inst[3]))), int(self.coerce_num(self.resolve_var(inst[4]))))
                        else:
                            iter_range = range(int(self.coerce_num(self.resolve_var(inst[3]))))

                        for i in iter_range:
                            self.write_var(inst[2], _tokenizer.token("number", i))
                            block_executer = _executer(code_block, self.global_vars, self.scope_str, self.schem_builder)
                            block_executer.macros = self.macros
                            block_executer.vars = self.vars
                            block_executer.allow_mlog = self.allow_mlog
                            block_executer.macro_run_counts = self.macro_run_counts
                            block_executer.execute()
                            self.output.extend(block_executer.output)
                case "log":
                    print("".join(map(self.resolve_log ,inst.tokens[1:-1])))
                case _:
                    for token in inst.tokens:
                        self.output.append(self.resolve_var(token))
            if len(self.output) > 0 and not self.allow_mlog:
                _error("Mlog instructions not allowed outside a 'proc' statement", inst[0])
            self.exec_pointer += 1
        if self.allow_mlog and self.is_root and len(self.output) > 0:
            self.schem_builder.add_proc(self.schem_builder.Proc(_tokenizer.token_list_to_str(self.output), None))
        if self.is_root:
            self.schem_builder.processor_type = self.global_vars["global_PROCESSOR_TYPE"]

    def check_for_proc(self) -> bool:
        for inst in self.read_lines():
            if inst[0].value in self.PROC_INSTRUCTIONS:
                break
        else:
            return False
        return True

    def read_till(self, end_word: str, start_word: list[str]) -> list[_tokenizer.token] | None: #None if eof, token if unexpected end
        lines = self.read_lines_till(end_word, start_word)
        if lines is None:
            return None
        else:
            return sum(lines, [])

    def read_lines_till(self, end_word: str, start_word: list[str]) -> list[list[_tokenizer.token]] | None:
        lines = []
        level = 0
        while True:
            self.exec_pointer += 1
            if self.exec_pointer >= len(self.instructions):
                return None
            inst = self.instructions[self.exec_pointer]
            if inst[0].value in start_word:
                level += 1
            elif inst[0].value == end_word and level > 0:
                level -= 1
            elif inst[0].value == end_word and level == 0:
                return lines
            lines.append(inst.tokens)

    def read_sections(self, end_word: str, start_word: list[str], split_word: list[str]) -> list[list[_tokenizer.token]] | None:
        sections = []
        section = []
        prev_line = self.instructions[self.exec_pointer]
        level = 0
        while True:
            self.exec_pointer += 1
            if self.exec_pointer >= len(self.instructions):
                return None
            inst = self.instructions[self.exec_pointer]
            if inst[0].value in start_word:
                level += 1
            elif inst[0].value == end_word and level > 0:
                level -= 1
            elif inst[0].value in split_word and level == 0:
                sections.append((prev_line, section))
                prev_line = inst
                section = []
            elif inst[0].value == end_word and level == 0:
                sections.append((prev_line, section))
                return sections
            else:
                section.extend(inst.tokens)

    def read_lines(self) -> list[list[_tokenizer.token]]:
        lines = []
        line = []
        for token in self.code:
            if token.type == "line_break":
                line.append(token)
                lines.append(self.InstructionLine(line))
                line = []
            else:
                line.append(token)
        return lines

    def as_root_level(self):
        self.allow_mlog = not self.check_for_proc()
        self.is_root = True
        self.global_vars["global_PROCESSOR_TYPE"] = _tokenizer.token("content_literal", "@micro-processor", 0, 0)

    def resolve_var(self, name: _tokenizer.token):
        if name.type == "identifier" and str(name) in self.vars:
            return self.vars[str(name)].with_scope(self.scope_str).at_pos((name.line, name.column))
        elif name.type == "global_identifier" and str(name) in self.global_vars:
            return self.global_vars[str(name)].with_scope("").at_pos((name.line, name.column))
        else:
            return name.with_scope(self.scope_str)

    def resolve_log(self, token: _tokenizer.token) -> str:
        if token.type == "string_literal":
            return token.value[1:-1]
        else:
            return str(self.resolve_var(token))

    def write_var(self, name: _tokenizer.token, value: _tokenizer.token):
        if name.type == "identifier":
            self.vars[str(name)] = value
        elif name.type == "global_identifier":
            self.global_vars[str(name)] = value
        else:
            return False
        return True

    def coerce_num(self, token: _tokenizer.token) -> float:
        if token.type == "number":
            return token.value
        elif token.type == "defined_literal" and token.value == "null":
            return 0
        elif token.type == "string_literal" and token.value == '""':
            return 0
        elif token.type in ["identifier", "global_identifier"]:
            return 0
        else:
            return 1

    def eval_math(self, operation: _tokenizer.token, input1: _tokenizer.token, input2: _tokenizer.token) -> _tokenizer.token:
        if input1.type == input2.type and operation.value in self.CONDITIONS:
            a = input1.value
            b = input2.value
        else:
            a = self.coerce_num(input1)
            b = self.coerce_num(input2)

        match operation.value:
            case "add":
                out = a + b
            case "sub":
                out = a - b
            case "mul":
                out = a * b
            case "div":
                out = a / b
            case "idiv":
                out = a // b
            case "mod":
                out = a % b
            case "pow":
                out = pow(a,b)
            case "equal":
                out = a == b
            case "notEqual":
                out = a != b
            case "land":
                out = a and b
            case "lessThan":
                out = a < b
            case "lessThanEq":
                out = a <= b
            case "greaterThan":
                out = a > b
            case "greaterThanEq":
                out = a >= b
            case "strictEqual":
                if input1.type != input2.type:
                    out = 0
                out = a == b
            case "shl":
                out = int(a) << int(b)
            case "shr":
                out = int(a) >> int(b)
            case "or":
                out = int(a) | int(b)
            case "and":
                out = int(a) & int(b)
            case "xor":
                out = int(a) ^ int(b)
            case "not":
                out = ~int(a)
            case "max":
                out = max(a, b)
            case "min":
                out = min(a, b)
            case "angle":
                out = math.degrees(math.atan2(a, b))
            case "angleDiff":
                a = ((a % 360) + 360) % 360
                b = ((b % 360) + 360) % 360
                out = min(a - b + 360 if (a - b) < 0 else a - b, b - a + 360 if (b - a) < 0 else b - a)
            case "len":
                out = math.hypot(a, b)
            case "abs":
                out = abs(a)
            case "log":
                out = math.log(a)
            case "log10":
                out = math.log10(a)
            case "floor":
                out = math.floor(a)
            case "ceil":
                out = math.ceil(a)
            case "sqrt":
                out = math.sqrt(a)
            case "rand":
                out = random.uniform(0,a)
            case "sin":
                out = math.sin(a)
            case "cos":
                out = math.cos(a)
            case "tan":
                out = math.tan(a)
            case "asin":
                out = math.asin(a)
            case "acos":
                out = math.acos(a)
            case "atan":
                out = math.atan(a)
            case _:
                _error(f"Unknown operation \"{operation.value}\"", operation)

        return _tokenizer.token("number", float(out))

    def eval_condition(self, operation: _tokenizer.token, input1: _tokenizer.token, input2: _tokenizer.token) -> _tokenizer.token:
        if input1.type == input2.type:
            a = input1.value
            b = input2.value
        else:
            a = self.coerce_num(input1)
            b = self.coerce_num(input2)      

        match operation.value:
            case "equal":
                out = a == b
            case "notEqual":
                out = a != b
            case "land":
                out = a and b
            case "lessThan":
                out = a < b
            case "lessThanEq":
                out = a <= b
            case "greaterThan":
                out = a > b
            case "greaterThanEq":
                out = a >= b
            case "strictEqual":
                if input1.type != input2.type:
                    out = 0
                out = a == b
            case _:
                _error(f"Unknown condition \"{operation.value}\"", operation)

        return _tokenizer.token("number", float(out))

class _schem_builder:
    class Proc:
        def __init__(self, code, pos):
            self.code: str = code
            self.pos = pos

    class Block:
        def __init__(self, inst: _tokenizer.token, type: _tokenizer.token, pos: tuple[int, int]|None, rot: int):
            self.inst = inst
            self.type_name = type.value[1:]
            self.type_token = type
            self.pos = pos
            self.rotation = rot
            self.link_name = ""

    def __init__(self):
        self.procs = []
        self.proc_positions = []
        self.placed_procs = []
        self.blocks = []
        self.link_counts = {}
        self.processor_type = None
        self.schem = pymsch.Schematic()

    def add_proc(self, proc):
        self.procs.append(proc)
        return f"processor{len(self.procs)}"

    def add_block(self, block):
        name = self.get_link_name(block.type_name)
        block.link_name = name
        self.blocks.append(block)
        return name

    def get_link_name(self, type: str):
        words = type.split('-')
        name = words[-2] if words[-1] == "large" else words[-1]
        if name in self.link_counts:
            self.link_counts[name] += 1
            name += str(self.link_counts[name])
        else:
            self.link_counts[name] = 1
            name += "1"
        return name

    def make_schem(self):
        self.schem_add_blocks()
        self.schem_add_procs()

    def schem_add_blocks(self):
        block_x = 0
        for block in self.blocks:
            for char in block.type_name:
                if char in "_ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    _error("Unknown block type", block.type_token)
            block_type_name = block.type_name.upper().replace('-', '_')
            if block_type_name in ["micro-processor", "logic-processor", "hyper-processor", "world-processor"]:
                _error("Block type must not be a processor, use 'proc'")
            if block_type_name not in pymsch.Content.__members__:
                _error("Unknown block type", block.type_token)
            block_type = pymsch.Content[block_type_name]
            
            if block.pos is None:
                while True:
                    new_block = self.schem.add_block(pymsch.Block(block_type, block_x, -(block_type.value.size//2) - 1, None, 0))
                    if new_block is not None:
                        block.pos = (block_x, -(block_type.value.size//2) - 1)
                        break
                    block_x += 1
                
            else:
                block = self.schem.add_block(pymsch.Block(block_type, block.pos[0], block.pos[1], None, block.rotation))
                if block is None:
                    _error("Specified position is blocked", block.inst)

    def schem_add_procs(self):
        if self.processor_type.value[1:] not in ["micro-processor", "logic-processor", "hyper-processor", "world-processor"]:
            _error("Unknown processor type", self.processor_type)
        proc_type = pymsch.Content[self.processor_type.value[1:].upper().replace('-', '_')]
        proc_size = proc_type.value.size
        square_size = math.ceil(math.sqrt(len(self.procs))) * proc_size
        while self.schem_count_filled_blocks(proc_size, square_size) + len(self.procs) > square_size**2:
            square_size += 1
        proc_x = math.ceil(proc_size/2) -1
        proc_y = math.ceil(proc_size/2) -1
        for proc in self.procs:
            while True:
                if proc_x >= square_size:
                    proc_x = math.ceil(proc_size/2) -1
                    proc_y += proc_size
                proc_conf = pymsch.ProcessorConfig(proc.code, [])
                block = self.schem.add_block(pymsch.Block(proc_type, proc_x, proc_y, proc_conf, 0))
                if block is None:
                    proc_x += proc_size
                else:
                    self.proc_positions.append((proc_x, proc_y))
                    self.placed_procs.append(block)
                    break
            proc_x += proc_size
        for proc in self.placed_procs:
            self.set_proc_links(proc.config, (proc.x, proc.y))

    def schem_count_filled_blocks(self, proc_size, square_size):
        count = 0
        for x in range(square_size):
            for y in range(square_size):
                inc = 0
                for px in range(proc_size):
                    for py in range(proc_size):
                        if (x*proc_size + px, y*proc_size + py) in self.schem._filled_list:
                            inc = 1
                            break
                count += inc
        return count

    def set_proc_links(self, proc, proc_pos):
        for block in self.blocks:
            proc.links.append(pymsch.ProcessorLink(block.pos[0] - proc_pos[0], block.pos[1] - proc_pos[1], block.link_name))
        for index, iter_proc in enumerate(self.proc_positions):
            proc.links.append(pymsch.ProcessorLink(iter_proc[0] - proc_pos[0], iter_proc[1] - proc_pos[1], f"processor{index+1}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='sfmlog', description='A mindustry transpiler', epilog=':hognar:')
    parser.add_argument('-s', '--src', required=True, type=pathlib.Path, help="the file to transpile", metavar="source_file")
    parser.add_argument('-o', '--out', type=pathlib.Path, help="the file to write the output to", metavar="output_file")
    parser.add_argument('-c', '--copy', action='store_true', help="copy the output to the clipboard")
    args = parser.parse_args()
    with open(args.src, 'r') as f:
        code = f.read()

    transpiler = SFMlog()
    out_schem = transpiler.transpile(code, args.src.parent)

    print(out_schem)

    if args.copy:
        out_schem.write_clipboard()