import sys, argparse, pathlib, re

def _error(text: str, token):
    print(f"ERROR at ({token.line},{token.column}): {text}")
    sys.exit(2)

class SFMlog:
    def __init__(self):
        pass

    def transpile(self, code: str, cwd: pathlib.Path) -> str:
        tokenizer = _tokenizer(code)
        parser = _parser(tokenizer.tokens, cwd)
        executer = _executer(parser.code, parser.macros, parser.functions, "")
        executer.execute()
        return _tokenizer.token_list_to_str(executer.output)

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
        def __init__(self, type: str, value, line: int, column: int, scope = None):
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

        if(string in ["true", "false", "null"]):
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
    class macro:
        def __init__(self, name: str, code: list[_tokenizer.token], args: list[str]):
            self.name = name
            self.code = code
            self.args = args
            self.run_count = 0

        def __repr__(self):
            return f"macro({self.name})"

    class function:
        def __init__(self, name: str, code: list[_tokenizer.token], args: list[str]):
            self.name = name
            self.code = code
            self.args = args

        def __repr__(self):
            return f"function({self.name})"

    def __init__(self, code: list[_tokenizer.token], cwd: pathlib.Path):
        self.cwd = cwd
        self.code = code
        self.macros = {}
        self.functions = {}
        self.imports = []

        self.parse()

    def parse(self):
        self.get_imports()
        self.get_macros()
        self.get_functions()

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

    def get_macros(self):
        code_iter = iter(self.code)
        code = []
        line = self.read_line(code_iter)
        while line != []:
            if self.list_get_token(line, 0).value == "defmac":
                mac_line = self.read_line(code_iter)
                mac_code = []
                while True:
                    if mac_line == []:
                        _error("Macro not closed", self.list_get_token(line, 0))
                    elif self.list_get_token(mac_line, 0).value == "endmac":
                        break
                    else:
                        mac_code.extend(mac_line)
                    mac_line = self.read_line(code_iter)
                macro = self.macro(self.list_get_token(line, 1).value, mac_code, line[2:-1])
                self.macros[macro.name] = macro
            elif self.list_get_token(line, 0).value == "endmac":
                _error("Unexpected endmac while not defining macro", self.list_get_token(line, 0))
            else:
                code.extend(line)
            line = self.read_line(code_iter)
        self.code = code

    def get_functions(self):
        code_iter = iter(self.code)
        code = []
        line = self.read_line(code_iter)
        while line != []:
            if self.list_get_token(line, 0).value == "deffun":
                fun_line = self.read_line(code_iter)
                fun_code = []
                while True:
                    if fun_line == []:
                        _error("Function not closed", self.list_get_token(line, 0))
                    elif self.list_get_token(fun_line, 0).value == "endfun":
                        break
                    else:
                        fun_code.extend(fun_line)
                    fun_line = self.read_line(code_iter)
                function = self.function(self.list_get_token(line, 1).value, fun_code, line[2:-1])
                self.functions[function.name] = function
            elif self.list_get_token(line, 0).value == "endfun":
                _error("Unexpected endfun while not defining function", self.list_get_token(line, 0))
            else:
                code.extend(line)
            line = self.read_line(code_iter)
        self.code = code

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
    def __init__(self, code: list[_tokenizer.token], macros: dict[str,_parser.macro], functions: dict[str,_parser.function], scope_str: str, called_functions: list[str] = None):
        self.code: list[_tokenizer.token] = code
        self.instructions = self.read_lines()
        self.output: list[_tokenizer.token] = []
        self.scope_str = scope_str
        self.macros: dict[str,_parser.macro] = macros
        self.functions: dict[str, _parser.function] = functions
        if called_functions is None:
            self.called_functions: list[str] = []
        else:
            self.called_functions: list[str] = called_functions
        self.vars: dict[str, _tokenizer.token] = {}

        self.exec_pointer = 0

    def execute(self):
        while True:
            if self.exec_pointer >= len(self.instructions):
                break
            inst = self.instructions[self.exec_pointer]

            match self.list_get_token(inst, 0).value:
                case "mac":
                    if self.list_get_token(inst, 1).value in self.macros:
                        mac = self.macros[self.list_get_token(inst, 1).value]
                        mac_executer = _executer(mac.code, self.macros, self.functions, f"_{mac.name}_{mac.run_count}_", self.called_functions)
                        for index, arg in enumerate(mac.args):
                            var_token = self.list_get_token(inst, index + 2)
                            if var_token.type in ["identifier", "global_identifier"] and str(var_token) in self.vars:
                                mac_executer.vars[str(arg)] = self.vars[str(var_token)]
                            else:
                                mac_executer.vars[str(arg)] = var_token.with_scope(self.scope_str)

                        mac.run_count += 1
                        mac_executer.execute()
                        self.output.extend(mac_executer.output)
                    else:
                        _error(f"Unknown macro '{self.list_get_token(inst, 1).value}'", self.list_get_token(inst, 1))
                #case "fun":
                #    function_name = self.list_get_token(inst, 1).value
                #    if function_name.value in self.functions:
                #        if function_name.value not in self.called_functions:
                #            self.called_functions.append(function_name.value)
                #        function = self.functions[function_name.value]
                #        for arg in function.args:
                #            self.extend([_parser.token("instruction", "set", ), ])
                #        ##type: str, value, line: int, column: int, scope = None
                #        #self.extend([_parser.token("instruction", "")])
                #    else:
                #        _error(f"Unknown function '{function_name.value}'", function_name)
                case "pset":
                    self.vars[str(self.list_get_token(inst, 1))] = self.list_get_token(inst, 2)
                case _:
                    for token in inst:
                        if str(token) in self.vars:
                            self.output.append(self.vars[str(token)].with_scope(self.scope_str).at_pos((token.line, token.column)))
                        else:
                            self.output.append(token.with_scope(self.scope_str))

            self.exec_pointer += 1

    def read_lines(self) -> list[list[_tokenizer.token]]:
        lines = []
        line = []
        for token in self.code:
            if token.type == "line_break":
                line.append(token)
                lines.append(line)
                line = []
            else:
                line.append(token)
        return lines

    def list_get_token(self, token_list: list[_tokenizer.token], index: int):
        try:
            token = token_list[index]
        except IndexError:
            _error("Unexpected end of line", token_list[-1])
        if token.type == "line_break":
            _error("Unexpected end of line", token)
        return token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='sfmlog', description='A mindustry transpiler', epilog=':hognar:')
    parser.add_argument('-s', '--src', required=True, type=pathlib.Path, help="the file to transpile", metavar="source_file")
    parser.add_argument('-o', '--out', type=pathlib.Path, help="the file to write the output to", metavar="output_file")
    parser.add_argument('-c', '--copy', action='store_true', help="copy the output to the clipboard")
    args = parser.parse_args()
    with open(args.src, 'r') as f:
        code = f.read()

    transpiler = SFMlog()
    out_code = transpiler.transpile(code, args.src.parent)

    print(out_code)

    if args.copy:
        import pyperclip
        pyperclip.copy(out_code)