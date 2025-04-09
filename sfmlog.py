import sys, argparse, pathlib, re, pymsch, math, random, time, copy, json

def _error(text: str, token):
    if token.file is None:
        print(f"ERROR at ({token.line},{token.column}): {text}")
    else:
        print(f"ERROR at ({token.line},{token.column}) in '{token.file}': {text}")
    sys.exit(2)

def _warning(text: str, token):
    if token.file is None:
        print(f"WARNING at ({token.line},{token.column}): {text}")
    else:
        print(f"WARNING at ({token.line},{token.column}) in '{token.file}': {text}")

class SFMlog:
    def __init__(self):
        pass

    def transpile(self, code: str, file: pathlib.Path) -> str:
        tokenizer = _tokenizer(code, file)
        schem_builder = _schem_builder()
        executer = _executer(tokenizer.tokens, file.parent, {}, "_", schem_builder)
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
        "while": [True],
        "for": [True]
    }

    LINK_BLOCKS = ["gate", "foundation", "wall", "container", "afflict", "heater", "conveyor", "duct", "press", "tower", "pad", "projector", "swarmer", "factory", "drill", "router", "door", "illuminator", "processor", "sorter", "spectre", "parallax", "cell", "electrolyzer", "display", "chamber", "mixer", "conduit", "distributor", "crucible", "message", "unloader", "refabricator", "switch", "bore", "bank", "accelerator", "disperse", "vault", "point", "nucleus", "panel", "node", "condenser", "smelter", "pump", "generator", "tank", "reactor", "cultivator", "malign", "synthesizer", "deconstructor", "meltdown", "centrifuge", "radar", "driver", "void", "junction", "diffuse", "pulverizer", "salvo", "bridge", "acropolis", "dome", "reconstructor", "separator", "citadel", "concentrator", "mender", "lancer", "source", "loader", "duo", "melter", "crusher", "fabricator", "redirector", "disassembler", "gigantic", "incinerator", "scorch", "battery", "tsunami", "arc", "compressor", "assembler", "smite", "module", "bastion", "segment", "constructor", "ripple", "furnace", "wave", "foreshadow", "link", "mine", "scathe", "canvas", "diode", "extractor", "fuse", "kiln", "sublimate", "scatter", "cyclone", "titan", "turret", "lustre", "thruster", "shard", "weaver", "huge", "breach", "hail"]

    class token:
        def __init__(self, type: str, value, line: int = 0, column: int = 0, file: pathlib.Path = None, scope = None, exportable = True):
            self.type: str = type
            self.value = value
            self.line = line
            self.column = column
            self.file = file
            self.scope = scope
            self.exportable = exportable

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
                return _tokenizer.token(self.type, self.value, self.line, self.column, self.file, scope=scope, exportable=self.exportable)
            else:
                return self

        def at_pos(self, pos: tuple[int, int]):
            return _tokenizer.token(self.type, self.value, pos[0], pos[1], self.file, scope=self.scope, exportable=self.exportable)

        def at_token(self, token):
            return _tokenizer.token(self.type, self.value, token.line, token.column, token.file , scope=self.scope, exportable=self.exportable)

        def resolve_string(self):
            if self.type == "string":
                return self.value[1:-2]
            else:
                return self.value

    def __init__(self, code: str, file: pathlib.Path):
        self.tokens: list[token] = self.tokenize(code, file)

    def tokenize(self, code: str, file: str) -> list[token]:
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
                    tokens.append(self.token(token_type, token_value, line + 1, column, file))
                prev_token_type = token_type
        tokens.append(self.token("line_break", "\n", line + 1, column, file))
        return tokens

    def identify_token(self, string: str, prev_token_type: str, prev_instruction: str, dist_from_prev_instruction: int, pos: tuple[int, int]) -> tuple[str, str | float]:
        if(string == '\n'):
            return ("line_break", "\n")

        if(string[0] == '"' and string[-1] == '"'):
            return ("string", string)

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
            return ("content", string)

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
            return ("null", string)

        return ("identifier", string)

    def token_list_to_str(tokens: list[token]) -> str:
        string = ""
        last_token = _tokenizer.token("line_break", "\n")
        for token in tokens:
            if token.type == "line_break" or last_token.type == "line_break":
                string += str(token)
            else:
                string += " " + str(token)
            last_token = token
        return string

class _executer:
    CONDITIONS = ["equal", "notEqual", "lessThan", "greaterThan", "lessThanEq", "greaterThanEq", "strictEqual"]
    DEFAULT_GLOBALS = {
        "PROCESSOR_TYPE":  _tokenizer.token("content", "@micro-processor"),
        "SCHEMATIC_NAME":  _tokenizer.token("string", '"SFMlog Schematic"'),
        "SCHEMATIC_DESCRIPTION": _tokenizer.token("string", '"This schematic was generated using SFMlog."')
    }

    class Instruction:
        def __init__(self, keyword, exec_func):
            self.keyword: str = keyword
            self.exec_func: Callable = exec_func

    class Instructions:
        def init_instructions(executer):
            inst = _executer.Instructions
            executer.init_instruction("import", inst.I_import)
            executer.init_instruction("block", inst.I_block)
            executer.init_instruction("proc", inst.I_proc)
            executer.init_instruction("defmac", inst.I_defmac)
            executer.init_instruction("mac", inst.I_mac)
            executer.init_instruction("getmac", inst.I_getmac)
            executer.init_instruction("setmac", inst.I_setmac)
            executer.init_instruction("type", inst.I_type)
            executer.init_instruction("pset", inst.I_pset)
            executer.init_instruction("pop", inst.I_pop)
            executer.init_instruction("strop", inst.I_strop)
            executer.init_instruction("strlabel", inst.I_strlabel)
            executer.init_instruction("strvar", inst.I_strvar)
            executer.init_instruction("list", inst.I_list)
            executer.init_instruction("table", inst.I_table)
            executer.init_instruction("if", inst.I_if)
            executer.init_instruction("while", inst.I_while)
            executer.init_instruction("for", inst.I_for)
            executer.init_instruction("log", inst.I_log)

        def I_import(inst, executer): # Imports and executes a separate sfmlog file
            import_file = executer.resolve_var(inst[1])
            if import_file.type == "string":
                import_file = pathlib.Path(import_file.value[1:-1])
            else:
                import_file = pathlib.Path(str(import_file.value))
            if not import_file.is_absolute():
                import_file = executer.cwd / import_file
            try:
                with open(import_file, "r") as file:
                    import_code = file.read()
            except FileNotFoundError:
                _error(f"File '{import_file}' not found", inst[1])
            import_tokenizer = _tokenizer(import_code, import_file)
            import_executer = _executer(import_tokenizer.tokens , import_file.parent, executer.global_vars, executer.scope_str, executer.schem_builder)
            import_executer.macros = executer.macros
            import_executer.vars = executer.vars
            import_executer.allow_mlog = executer.allow_mlog
            import_executer.macro_run_counts = executer.macro_run_counts
            import_executer.execute()
            executer.output.extend(import_executer.output)

        def I_block(inst, executer): # Adds a block to the schematic
            var_name = inst[1]
            if var_name.type not in ["identifier", "global_identifier"]:
                _error("Invalid variable name", var_name)
            block_type = inst[2]
            if block_type.type != "content":
                _error("Expected block type", block_type)
            block_pos = None
            block_rot = 0
            if 4 in inst:
                if executer.resolve_var(inst[3]).type != "number":
                    _error("Expected numeric value", inst[3])
                if executer.resolve_var(inst[4]).type != "number":
                    _error("Expected numeric value", inst[4])
                block_pos = (int(executer.resolve_var(inst[3]).value), int(executer.resolve_var(inst[4]).value))
            if 5 in inst:
                if type(executer.resolve_var(inst[5]).value) != float:
                    _error("Expected numeric value", inst[5])
                block_rot = int(executer.resolve_var(inst[5]).value)

            block = executer.schem_builder.Block(inst, block_type, block_pos, block_rot)
            link_name = executer.schem_builder.add_block(block)
            executer.write_var(var_name, _tokenizer.token("block", link_name))

        def I_proc(inst, executer): # Adds a processor to the schematic
            proc_code = executer.read_till("endproc", ["proc"])
            if proc_code is None:
                _error("'endproc' expected, but not found", inst[0])
            proc_executer = _executer(proc_code, executer.cwd, executer.global_vars, "_", executer.schem_builder)
            proc_executer.macros = executer.macros
            proc_executer.execute()
            if 4 in inst:
                proc_type = executer.resolve_var(inst[2])
            elif 2 in inst:
                _error("Unable to define type of proc without defined position", inst[2])
            else:
                proc_type = None
            if 4 in inst:
                if executer.resolve_var(inst[3]).type != "number":
                    _error("Expected numeric value", inst[3])
                if executer.resolve_var(inst[4]).type != "number":
                    _error("Expected numeric value", inst[4])
                pos = (int(executer.resolve_var(inst[3]).value), int(executer.resolve_var(inst[4]).value))
            else:
                pos = None
            proc_name = executer.schem_builder.add_proc(executer.schem_builder.Proc(_tokenizer.token_list_to_str(proc_executer.output), pos, proc_type, inst))
            if 1 in inst:
                executer.write_var(inst[1], _tokenizer.token("block", proc_name))

        def I_defmac(inst, executer): # Defines a macro
            mac_code = executer.read_till("endmac", ["defmac"])
            if mac_code is None:
                _error("'endmac' expected, but not found", inst[0])
            if inst[1].type != "identifier":
                _error("Invalid name for macro", inst[1])
            mac_args = []
            for arg in inst.tokens[2:-1]:
                if arg.type != "identifier":
                    _error("Invalid name for macro argument", arg)
                mac_args.append(arg)
            executer.macros[inst[1].value] = executer.Macro(inst[1].value, mac_code, mac_args)

        def I_mac(inst, executer): # Calls a macro
            if inst[1].value in executer.macros:
                mac = executer.macros[inst[1].value]
                if mac.name not in executer.macro_run_counts:
                    executer.macro_run_counts[mac.name] = 0
                mac_executer = _executer(mac.code, executer.cwd, executer.global_vars, f"{executer.scope_str}{mac.name}_{executer.macro_run_counts[mac.name]}_", executer.schem_builder)
                for index, arg in enumerate(mac.args):
                    var_token = inst[index + 2]
                    mac_executer.write_var(arg, executer.resolve_var(var_token))
                mac_executer.macros = executer.macros.copy()

                executer.macro_run_counts[mac.name] += 1
                mac_executer.execute()
                executer.output.extend(mac_executer.output)
                for index, arg in enumerate(mac.args):
                    var_token = inst[index + 2]
                    executer.write_var(var_token, mac_executer.resolve_var(arg))
            else:
                _error(f"Unknown macro '{inst[1].value}'", inst[1])
            
        def I_getmac(inst, executer): # Writes a macro to a variable
            if inst[2].value not in executer.macros:
                _error(f"Unknown macro '{inst[2].value}'", inst[2])
            executer.write_var(inst[1], executer.convert_to_var(executer.macros[inst[2].value]))

        def I_setmac(inst, executer): # Sets a macro from a variable
            mac = executer.resolve_var(inst[2])
            if mac.type != "macro":
                _error(f"Variable '{inst[2].value}' isn't of type 'macro'", inst[2])
            if inst[1].type != "identifier":
                _error("Invalid name for macro", inst[1])
            executer.macros[inst[1].value] = mac.value

        def I_type(inst, executer): # Gets the type of a value
            executer.write_var(inst[1], executer.convert_to_var(executer.resolve_var(inst[2]).type))

        def I_pset(inst, executer): # Sets a variable
            value = executer.resolve_var(inst[2])
            if value.type in ["identifier", "global_identifier"]:
                _error(f"Unable to write type '{value.type}' to a variable", inst[2])
            executer.write_var(inst[1], value)

        def I_pop(inst, executer): # Performs math operations
            executer.write_var(inst[2], executer.eval_math(inst[1], executer.resolve_var(inst[3]), executer.resolve_var(inst[4])))

        def I_strop(inst, executer): # Performs string operations ## TODO
            str_op = inst[1]
            str_out = inst[2]
            str_in = str(executer.convert_var_to_py(executer.resolve_var(inst[3])))
            str_in2 = str(executer.convert_var_to_py(executer.resolve_var(inst[4])))
            out_val = ""
            match str_op.value:
                case "cat":
                    out_val = str_in + str_in2
            executer.write_var(str_out, executer.convert_to_var(out_val))

        def I_strlabel(inst, executer): # Creates a label from a string
            value = executer.resolve_var(inst[1])
            if value.type != "string":
                _error(f"Expected type 'string', got type '{value.type}'", inst[1])
            executer.output.append(_tokenizer.token("label", value.value[1:-1].replace(" ", "_") + ':').with_scope(executer.scope_str).at_token(inst[1]))
            executer.output.append(inst.tokens[-1])

        def I_strvar(inst, executer): # Writes a variable name to a variable from a string ## TODO
            var_out = inst[2]
            str_in = executer.resolve_var(inst[3])
            if str_in.type != "string":
                _error(f"Expected type 'string', got type '{str_in.type}'", inst[3])

            token_type = None
            match inst[1].value:
                case "local":
                    token_type = "identifier"
                case "global":
                    token_type = "global_identifier"
                case _:
                    _error(f"Unknown variable context '{inst[1].value}'")

            executer.write_var(var_out, _tokenizer.token(token_type, str_in.value[1:-1].replace(" ", "_")))

        def I_list(inst, executer): # Performs list operations
            match inst[1].value:
                case "from": # Creates a list from instruction arguments
                    output_list = inst[2]
                    lst = []
                    for elem in inst.tokens[3:-1]:
                        value = executer.resolve_var(elem)
                        lst.append(value)
                    executer.write_var(output_list, executer.convert_to_var(lst))
                case "set": # Sets an index value
                    output_list = inst[2]
                    input_list = inst[3]
                    if input_list.type in ["identifier", "global_identifier"]:
                        var = executer.resolve_var(input_list)
                        lst = copy.deepcopy(var.value) if var.type == "list" else []
                    else:
                        lst = []
                    value = executer.resolve_var(inst[4])
                    index = executer.resolve_var(inst[5])
                    if index.type != "number":
                        _error(f"Expected type 'number', got type '{index.type}'", inst[5])
                    try:
                        lst[int(index.value)] = value
                    except IndexError:
                        _error("Index out of range", inst[5])
                    executer.write_var(output_list, executer.convert_to_var(lst))
                case "get": # Gets an index value
                    output = inst[2]
                    input_list = inst[3]
                    if input_list.type in ["identifier", "global_identifier"]:
                        var = executer.resolve_var(input_list)
                        lst = copy.deepcopy(var.value) if var.type == "list" else []
                    else:
                        lst = []
                    index = executer.resolve_var(inst[4])
                    if index.type != "number":
                        _error(f"Expected type 'number', got type '{index.type}'", inst[4])
                    try:
                        executer.write_var(output, lst[int(index.value)])
                    except IndexError:
                        _error("Index out of range", inst[4])
                case "append": # Appends value to the end
                    output_list = inst[2]
                    input_list = inst[3]
                    if input_list.type in ["identifier", "global_identifier"]:
                        var = executer.resolve_var(input_list)
                        lst = copy.deepcopy(var.value) if var.type == "list" else []
                    else:
                        lst = []
                    value = executer.resolve_var(inst[4])
                    lst.append(value)
                    executer.write_var(output_list, executer.convert_to_var(lst))
                case "insert": # Inserts value at an index
                    output_list = inst[2]
                    input_list = inst[3]
                    if input_list.type in ["identifier", "global_identifier"]:
                        var = executer.resolve_var(input_list)
                        lst = copy.deepcopy(var.value) if var.type == "list" else []
                    else:
                        lst = []
                    value = executer.resolve_var(inst[4])
                    index = executer.resolve_var(inst[5])
                    if index.type != "number":
                        _error(f"Expected type 'number', got type '{index.type}'", inst[5])
                    lst.insert(int(index.value), value)
                    executer.write_var(output_list, executer.convert_to_var(lst))
                case "del": # Removes value from an index
                    output_list = inst[2]
                    input_list = inst[3]
                    if input_list.type in ["identifier", "global_identifier"]:
                        var = executer.resolve_var(input_list)
                        lst = copy.deepcopy(var.value) if var.type == "list" else []
                    else:
                        lst = []
                    index = executer.resolve_var(inst[4])
                    if index.type != "number":
                        _error(f"Expected type 'number', got type '{index.type}'", inst[4])
                    try:
                        lst.pop(int(index.value))
                    except IndexError:
                        _error("Index out of range", inst[4])
                    executer.write_var(output_list, executer.convert_to_var(lst))
                case "len": # Gets length
                    output = inst[2]
                    input_list = inst[3]
                    if executer.resolve_var(input_list).type == "list":
                        executer.write_var(output, executer.convert_to_var(len(executer.resolve_var(input_list).value)))
                    else:
                        executer.write_var(output, executer.convert_to_var(None))
                case "index": # Gets the index of an item
                    output = inst[2]
                    input_list = inst[3]
                    input_elem = executer.resolve_var(inst[4])
                    if input_list.type in ["identifier", "global_identifier"]:
                        var = executer.resolve_var(input_list)
                        lst = copy.deepcopy(var.value) if var.type == "list" else []
                    else:
                        lst = []
                    if executer.resolve_var(input_list).type == "list":
                        for index, elem in enumerate(lst):
                            if elem.type == input_elem.type and elem.value == input_elem.value:
                                executer.write_var(output, executer.convert_to_var(index))
                                break
                        else:
                            executer.write_var(output, executer.convert_to_var(-1))
                    else:
                        executer.write_var(output, executer.convert_to_var(None))
                case "in": # Checks if item is in list
                    output = inst[2]
                    input_list = inst[3]
                    input_elem = executer.resolve_var(inst[4])
                    if input_list.type in ["identifier", "global_identifier"]:
                        var = executer.resolve_var(input_list)
                        lst = copy.deepcopy(var.value) if var.type == "list" else []
                    else:
                        lst = []
                    if executer.resolve_var(input_list).type == "list":
                        for elem in lst:
                            if elem.type == input_elem.type and elem.value == input_elem.value:
                                executer.write_var(output, executer.convert_to_var(1))
                                break
                        else:
                            executer.write_var(output, executer.convert_to_var(0))
                    else:
                        executer.write_var(output, executer.convert_to_var(None))
                case _:
                    _error(f"Unknown list operation \"{inst[1].value}\"", inst[1])
        
        def I_table(inst, executer): # Performs table operations
            match inst[1].value:
                case "from": # Creates a table from sequential key value pairs
                    output_table = inst[2]
                    tbl = {}
                    if len(inst.tokens[3:-1]) % 2 != 0:
                        _error("Unfinished key value pair", inst.tokens[-1])
                    for i in range(len(inst.tokens[3:-1])//2):
                        elem1 = inst.tokens[(i*2)+3]
                        elem2 = inst.tokens[(i*2)+4]
                        key = executer.resolve_var(elem1)
                        value = executer.resolve_var(elem2)
                        if key.type in ["list", "table"]:
                            _error(f"Unable to write type '{key.type}' to table key", elem1)
                        tbl[executer.convert_var_to_py(key)] = value
                    executer.write_var(output_table, executer.convert_to_var(tbl))
                case "set": # Sets a key's value in a table
                    output_table = inst[2]
                    input_table = inst[3]
                    if input_table.type in ["identifier", "global_identifier"]:
                        var = executer.resolve_var(input_table)
                        tbl = copy.deepcopy(var.value) if var.type == "table" else {}
                    else:
                        tbl = {}
                    key = executer.resolve_var(inst[4])
                    value = executer.resolve_var(inst[5])
                    if key.type in ["list", "table"]:
                        _error(f"Unable to write type '{key.type}' to table key", inst[4])
                    tbl[executer.convert_var_to_py(key)] = value
                    executer.write_var(output_table, executer.convert_to_var(tbl))
                case "get": # Gets a key's value in a table
                    output = inst[2]
                    input_table = inst[3]
                    if input_table.type in ["identifier", "global_identifier"]:
                        var = executer.resolve_var(input_table)
                        tbl = copy.deepcopy(var.value) if var.type == "table" else {}
                    else:
                        tbl = {}
                    key = executer.resolve_var(inst[4])
                    try:
                        executer.write_var(output, tbl[executer.convert_var_to_py(key)])
                    except KeyError:
                        _error(f"Key '{key.value}' not found", inst[4])
                case "del": # Removes a key
                    output_table = inst[2]
                    input_table = inst[3]
                    if input_table.type in ["identifier", "global_identifier"]:
                        var = executer.resolve_var(input_table)
                        tbl = copy.deepcopy(var.value) if var.type == "table" else {}
                    else:
                        tbl = {}
                    key = executer.resolve_var(inst[4])
                    try:
                        tbl.pop(key.value)
                    except KeyError:
                        _error(f"Key '{key.value}' not found", inst[4])
                    executer.write_var(output_table, executer.convert_to_var(tbl))
                # json excluded for now because it's basically useless for sfmlog
                #case "readjson": # Creates a table from a json string
                #    output_table = inst[2]
                #    input_str = executer.resolve_var(inst[3])
                #    if input_str.type == "string":
                #        executer.write_var(output_table, executer.convert_to_var(json.loads(input_str.value[1:-1]), expand_strings=True))
                #case "writejson": # Creates a json string from a table
                #    output_str = inst[2]
                #    input_table = inst[3]
                #    if input_table.type in ["identifier", "global_identifier"]:
                #        var = executer.resolve_var(input_table)
                #        tbl = copy.deepcopy(var.value) if var.type == "table" else {}
                #    else:
                #        tbl = {}
                #    executer.write_var(output_str, executer.convert_to_var(json.dumps(executer.convert_var_to_py(executer.resolve_var(input_table)))))
                case _:
                    _error(f"Unknown table operation \"{inst[1].value}\"", inst[1]) 
        
        def I_if(inst, executer): # Runs code depending on a condition
            code_sections = executer.read_sections("endif", ["if"], ["elif", "else"])
            if code_sections is None:
                _error("'endif' expected, but not found", inst[0])

            for instruction, code_block in code_sections:
                if instruction[0].value == "else" or executer.eval_condition(instruction[1], executer.resolve_var(instruction[2]), executer.resolve_var(instruction.option(3))).value:
                    block_executer = _executer(code_block, executer.cwd, executer.global_vars, executer.scope_str, executer.schem_builder)
                    block_executer.macros = executer.macros
                    block_executer.vars = executer.vars
                    block_executer.allow_mlog = executer.allow_mlog
                    block_executer.macro_run_counts = executer.macro_run_counts
                    block_executer.execute()
                    executer.output.extend(block_executer.output)
                    break
        
        def I_while(inst, executer): # Loops code depending on a condition
            code_block = executer.read_till("endwhile", ["while"])
            if code_block is None:
                _error("'endwhile' expected, but not found", inst[0])
            while executer.eval_condition(inst[1], executer.resolve_var(inst[2]), executer.resolve_var(inst.option(3))).value:
                block_executer = _executer(code_block, executer.cwd, executer.global_vars, executer.scope_str, executer.schem_builder)
                block_executer.macros = executer.macros
                block_executer.vars = executer.vars
                block_executer.allow_mlog = executer.allow_mlog
                block_executer.macro_run_counts = executer.macro_run_counts
                block_executer.execute()
                executer.output.extend(block_executer.output)
        
        def I_for(inst, executer): # Loops code via iterator operations
            code_block = executer.read_till("endfor", ["for"])
            if code_block is None:
                _error("'endfor' expected, but not found", inst[0])
            for_iter = None
            match inst[1].value:
                case "range":
                    if 5 in inst:
                        if int(executer.coerce_num(executer.resolve_var(inst[5]))) == 0:
                            _error("'for range' step value must not be zero", inst[5])
                        for_iter = range(int(executer.coerce_num(executer.resolve_var(inst[3]))), int(executer.coerce_num(executer.resolve_var(inst[4]))), int(executer.coerce_num(executer.resolve_var(inst[5]))))
                    elif 4 in inst:
                        for_iter = range(int(executer.coerce_num(executer.resolve_var(inst[3]))), int(executer.coerce_num(executer.resolve_var(inst[4]))))
                    else:
                        for_iter = range(int(executer.coerce_num(executer.resolve_var(inst[3]))))
                case "list":
                    lst = executer.resolve_var(inst[3])
                    if lst.type != "list":
                        _error(f"Expected type 'list', got '{lst.type}'", inst[3])
                    for_iter = iter(lst.value)
                case "table":
                    tbl = executer.resolve_var(inst[4])
                    if tbl.type != "table":
                        _error(f"Expected type 'table', got '{tbl.type}'", inst[4])
                    for_iter = tbl.value.items()

            for i in for_iter:
                if type(i) == tuple:
                    for index, value in enumerate(i):
                        executer.write_var(inst[2+index], executer.convert_to_var(value))
                else:
                    executer.write_var(inst[2], executer.convert_to_var(i))
                block_executer = _executer(code_block, executer.cwd, executer.global_vars, executer.scope_str, executer.schem_builder)
                block_executer.macros = executer.macros
                block_executer.vars = executer.vars
                block_executer.allow_mlog = executer.allow_mlog
                block_executer.macro_run_counts = executer.macro_run_counts
                block_executer.execute()
                executer.output.extend(block_executer.output)
        
        def I_log(inst, executer): # Writes out to the console
            print("".join(map(executer.resolve_log ,inst.tokens[1:-1])))

    class InstructionLine:
        def __init__(self, tokens):
            self.tokens = tokens

        def require(self, index: int):
            try:
                out = self.tokens[index]
            except IndexError:
                _error(f"Instruction '{self.tokens[0].value}' expected argument at position {index}", self.tokens[-1])
            if out.type == "line_break":
                _error(f"Instruction '{self.tokens[0].value}' expected argument at position {index}", out)
            return out

        def option(self, index, default = None):
            if default is None:
                default = _tokenizer.token("null", "null").at_token(self.tokens[-1])
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

        def __str__(self):
            return f"macro({self.name})"

    def __init__(self, code: list[_tokenizer.token], cwd: pathlib.Path, global_vars: dict[str, _tokenizer.token], scope_str: str, schem_builder):
        self.instructions: list[Instruction] = []
        self.Instructions.init_instructions(self)
        self.code: list[_tokenizer.token] = code
        self.lines = self.read_lines()
        self.output: list[_tokenizer.token] = []
        self.cwd: pathlib.Path = cwd
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
            if self.exec_pointer >= len(self.lines):
                break
            inst = self.lines[self.exec_pointer]
            
            self.exec_instruction(inst)

            if len(self.output) > 0 and not self.allow_mlog:
                _error("Mlog instructions not allowed outside a 'proc' statement", inst[0])
            self.exec_pointer += 1
        if self.allow_mlog and self.is_root and len(self.output) > 0:
            self.schem_builder.add_proc(self.schem_builder.Proc(_tokenizer.token_list_to_str(self.output), None, self.global_vars["global_PROCESSOR_TYPE"], None))
        if self.is_root:
            self.schem_builder.processor_type = self.global_vars["global_PROCESSOR_TYPE"]
            self.schem_builder.set_name(self.resolve_log(self.global_vars["global_SCHEMATIC_NAME"]))
            self.schem_builder.set_desc(self.resolve_log(self.global_vars["global_SCHEMATIC_DESCRIPTION"]))

    def check_for_proc(self) -> bool:
        for inst in self.read_lines():
            if inst[0].value == "proc":
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
            if self.exec_pointer >= len(self.lines):
                return None
            inst = self.lines[self.exec_pointer]
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
        prev_line = self.lines[self.exec_pointer]
        level = 0
        while True:
            self.exec_pointer += 1
            if self.exec_pointer >= len(self.lines):
                return None
            inst = self.lines[self.exec_pointer]
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
        for name, value in self.DEFAULT_GLOBALS.items():
            self.global_vars[f"global_{name}"] = value

    def convert_to_var(self, value):
        match value:
            case (int() | float() | bool()):
                return _tokenizer.token("number", float(value))
            case str() as v if v[0] == '"' and v[-1] == '"':
                return _tokenizer.token("string", value)
                return _tokenizer.token("string", value.replace("\\\"", "\""))
            case str() as v if v[0] == '@':
                return _tokenizer.token("content", value)
            case str():
                return _tokenizer.token("string", '"' + value + '"' )
            case list():
                lst = []
                for item in value:
                    lst.append(self.convert_to_var(item))
                return _tokenizer.token("list", lst, exportable = False)
            case dict():
                tbl = {}
                for key, value in value.items():
                    tbl[self.convert_var_to_py(self.convert_to_var(key))] = self.convert_to_var(value)
                return _tokenizer.token("table", tbl, exportable = False)
            case self.Macro():
                return _tokenizer.token("macro", value, exportable = False)
            case None:
                return _tokenizer.token("null", "null")
            case _tokenizer.token():
                return value
            case _:
                raise Exception(f"Unhandled type '{type(value)}'")

    def convert_var_to_py(self, var):
        match var.type:
            case ("number"|"content"|"identifier"|"global_identifier"|"block"):
                return var.value
            case "string":
                return var.value[1:-1]
            case ("null"|"macro"):
                return None
            case "list":
                return [self.convert_var_to_py(x) for x in var.value]
            case "table":
                return {k: self.convert_var_to_py(v) for k, v in var.value.items()}
            case _:
                raise Exception(f"Unable to convert type '{var.type}'")

    def resolve_var(self, name: _tokenizer.token):
        if name.type == "identifier" and str(name) in self.vars:
            return self.vars[str(name)].with_scope(self.scope_str).at_token(name)
        elif name.type == "global_identifier" and str(name) in self.global_vars:
            return self.global_vars[str(name)].with_scope("").at_token(name)
        else:
            return name.with_scope(self.scope_str)

    def resolve_log(self, token: _tokenizer.token) -> str:
        if self.resolve_var(token).type == "string":
            return self.resolve_var(token).value[1:-1].replace("\\n", "\n")
        elif self.resolve_var(token).type == "list":
            return f'[{", ".join([self.resolve_output(x) for x in self.resolve_var(token).value])}]'
        elif self.resolve_var(token).type == "table":
            return f'{{{", ".join([f"{str(k)}: {self.resolve_output(v)}" for k, v in self.resolve_var(token).value.items()])}}}'
        else:
            return str(self.resolve_var(token))

    def resolve_output(self, token: _tokenizer.token) -> str:
        if self.resolve_var(token).type == "list":
            return f'[{", ".join([self.resolve_output(x) for x in self.resolve_var(token).value])}]'
        elif self.resolve_var(token).type == "table":
            return f'{{{", ".join([f"{str(k)}: {self.resolve_output(v)}" for k, v in self.resolve_var(token).value.items()])}}}'
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
        elif token.type == "null" and token.value == "null":
            return 0
        elif token.type == "string" and token.value == '""':
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
                else:
                    out = a == b
            case "in":
                if input1.type != "list":
                    out = 0
                else:
                    for elem in input1.value:
                        if elem.type == input2.type and elem.value == input2.value:
                            out = 1
                            break
                    else:
                        out = 0
            case _:
                _error(f"Unknown condition \"{operation.value}\"", operation)

        return self.convert_to_var(out)

    def init_instruction(self, keyword, exec_func):
        instruction = self.Instruction(keyword, exec_func)
        self.instructions.append(instruction)

    def exec_instruction(self, inst):
        for i in self.instructions:
            if inst[0].value == i.keyword:
                i.exec_func(inst, self)
                break
        else:
            for token in inst.tokens:
                if self.resolve_var(token).exportable:
                    self.output.append(self.resolve_var(token))
                else:
                    _error(f"Unable to output type '{self.resolve_var(token).type}' to mlog", token)

class _schem_builder:
    class Proc:
        def __init__(self, code, pos, proc_type, inst):
            self.code: str = code
            self.pos = pos
            self.type = proc_type
            self.inst = inst

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

    def set_name(self, name):
        self.schem.set_tag('name', name)

    def set_desc(self, desc):
        self.schem.set_tag('description', desc)

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
                placed_block = self.schem.add_block(pymsch.Block(block_type, block.pos[0], block.pos[1], None, block.rotation))
                if placed_block is None:
                    _warning(f"Specified position at {block.pos} is blocked", block.inst[0])

    def schem_add_positioned_procs(self):
        procs = [x for x in self.procs if x.pos is not None]
        for proc in procs:
            if proc.type.value[1:] not in ["micro-processor", "logic-processor", "hyper-processor", "world-processor"]:
                _error("Unknown processor type", proc.type)
            proc_type = pymsch.Content[proc.type.value[1:].upper().replace('-', '_')]
            proc_conf = pymsch.ProcessorConfig(proc.code, [])
            block = self.schem.add_block(pymsch.Block(proc_type, proc.pos[0], proc.pos[1], proc_conf, 0))
            if block is not None:
                self.proc_positions.append(proc.pos)
                self.placed_procs.append(block)
            else:
                _warning(f"Specified position at {proc.pos} is blocked", proc.inst[0])

    def schem_add_unpositioned_procs(self):
        procs = [x for x in self.procs if x.pos is None]
        if self.processor_type.value[1:] not in ["micro-processor", "logic-processor", "hyper-processor", "world-processor"]:
            _error("Unknown processor type", self.processor_type)
        proc_type = pymsch.Content[self.processor_type.value[1:].upper().replace('-', '_')]
        proc_size = proc_type.value.size
        square_size = math.ceil(math.sqrt(len(procs))) * proc_size
        while self.schem_count_filled_blocks(proc_size, square_size) + len(procs) > square_size**2:
            square_size += 1
        proc_x = math.ceil(proc_size/2) -1
        proc_y = math.ceil(proc_size/2) -1
        for proc in procs:
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

    def schem_add_procs(self):
        self.schem_add_positioned_procs()
        self.schem_add_unpositioned_procs()
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
    start_time = time.perf_counter()
    out_schem = transpiler.transpile(code, args.src)
    end_time = time.perf_counter()

    print(f"Created schematic '{out_schem.tags["name"]}' in {end_time - start_time:0.2f} seconds" )

    if args.copy:
        out_schem.write_clipboard()