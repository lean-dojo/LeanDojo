import system.io

open expr
open tactic
open interaction_monad interaction_monad.result
open lean
open lean.parser
open native

universe u

namespace lean_dojo

namespace io

meta def put_str_ln' : Π (fmt : format), io unit := io.put_str_ln ∘ format.to_string

meta def fail' {α} (fmt : format) : io α := io.fail $ format.to_string fmt

/-- Version of io.run_tactic which does not suppress the exception msg -/
meta def run_tactic'' {α} (tac : tactic α) : io α := do {
  io.run_tactic $ do {
    result ← tactic.capture tac,
    match result with
    | (success val _) := pure val
    | (exception m_fmt _ _) := do {
      let fmt_msg := (m_fmt.get_or_else (λ _, format!"n/a")) (),
      let msg := format!"[fatal] {fmt_msg}",
      tactic.fail msg
    }
    end
  }
}

end io

namespace interaction_monad

meta def run_with_state' {σ₁ σ₂ : Type} {α : Type*} (state : σ₁) (tac : interaction_monad σ₁ α) : interaction_monad σ₂ α :=
λ s, match (tac state) with
     | (success val _) := success val s
     | (exception fn pos _) := exception fn pos s
     end
     
end interaction_monad

namespace expr

/--
Returns true if `e` contains a `sorry`.
See also `name.contains_sorry`.
-/
meta def contains_sorry (e : expr) : bool :=
e.fold ff (λ e' _ b, if (is_sorry e').is_some then tt else b)

meta def app_symbol_is (e : expr) (nm : name) : bool :=
match e.get_app_fn with
| (expr.const n _) := n = nm
| _ := ff
end

meta def contains_undefined (e : expr) : bool :=
e.fold ff $ λ e' _ b, if app_symbol_is e' `undefined then tt else b

end expr

namespace tactic

/- capture but backtrack the state -/
meta def capture' {α} (t : tactic α) : tactic (tactic_result α) :=
λ s, match t s with
| (success r s') := success (success r s') s
| (exception f p s') := success (exception f p s') s
end

meta def guard_sorry (e : expr) : tactic unit := guard $ bnot (expr.contains_sorry e)

meta def guard_undefined (e : expr) : tactic unit := guard $ bnot (expr.contains_undefined e)

end tactic

meta def kernel_type_check (pf : expr) : tactic unit := do {
  tp ← tactic.infer_type pf,
  env ← tactic.get_env,
  let decl := (declaration.defn `_ (expr.collect_univ_params pf) tp pf reducibility_hints.opaque ff),
  res ← tactic.capture' (env.add decl $> ()),
  match res with
  | (interaction_monad.result.success _ _) := pure ()
  | (interaction_monad.result.exception msg _ _) := let msg := msg.get_or_else (λ _, ("" : format)) in
    tactic.fail format! "kernel type check failed:\n---\n{msg ()}\n---\n"
  end
}

meta def validate_proof (tgt: expr) (pf: expr) : tactic unit := do {
    env ← tactic.get_env,
    pf ← pure $ env.unfold_untrusted_macros pf,
    pft ← tactic.infer_type pf,
    tactic.type_check pf tactic.transparency.all,
    guard (bnot pf.has_meta_var) <|> do {
      tactic.fail format! "proof contains metavariables"
    },
    tactic.guard_sorry pf <|> do {
      tactic.fail format! "proof contains `sorry`"
    },
    tactic.guard_undefined pf <|> do {
      tactic.fail format! "proof contains `undefined`"
    },
    tactic.is_def_eq tgt pft <|> do {
      tgt_fmt ← tactic.pp tgt,
      pft_fmt ← tactic.pp pft,
      tactic.fail format! "proof type mismatch: {tgt_fmt} != {pft_fmt}"
    },
    kernel_type_check pf
}

/-- `get_state` returns the underlying state inside an interaction monad, from within that monad. -/
-- Note that this is a generalization of `tactic.read` in core.
meta def get_state {σ : Type} : interaction_monad σ σ :=
λ state, success state state

/-- Run the given parser on the given string input. -/
meta def run_on_input {α} (p : lean.parser α) (s : string) : tactic α :=
lean.parser.run $ do
  get_state >>= λ ps, of_tactic $ do
    tactic.set_env ps.env,
    -- eval_trace format!"[parse_itactic_reflected] TRYING TO PARSE {itactic_string}",
    prod.fst <$> (@interaction_monad.run_with_state' parser_state _ _ ps $ with_input p s)

/-- Parse a reflected interactive tactic from a string.
    The result can be evaluated to a `tactic unit` by using
    `eval_expr (tactic unit)`. -/
meta def parse_itactic_reflected (tactic_string : string) : tactic expr := do
let itactic_string := "{ " ++ tactic_string ++  " }",
r ← run_on_input parser.itactic_reflected itactic_string,
pure $ reflected_value.expr r

/-- Parse an interactive tactic from a string. -/
meta def parse_itactic (tactic_string : string) : tactic (tactic string) :=
do
  rtac ← parse_itactic_reflected tactic_string,
  u ← eval_expr (tactic unit) rtac,
  pure (u *> pure tactic_string)

meta def get_tac_and_capture_result (next_candidate : string) (timeout : nat) : tactic (tactic_result _) := do {
  tac ← do {
    env ← tactic.get_env,
    tac ← parse_itactic next_candidate,
    tactic.set_env env,
    pure tac
  },
  result ← tactic.capture' (tactic.try_for_time timeout $ tactic.try_for 200000 tac), -- if `tac` fails, exception is captured here
  pure result
}

meta structure LeanREPLRequest : Type :=
(cmd : string)
(tsid: option nat)
(tac: option string)
(name: option string)

meta class has_from_json (α : Type u) : Type (u+1) :=
(from_json : json → tactic α)

meta instance : has_from_json LeanREPLRequest := ⟨λ msg, match msg with
  | (json.array [json.of_string cmd, json.array args]) := match cmd with
    | "run_tac" := match json.array args with
      | (json.array [json.of_int (int.of_nat tsid), json.of_string tac]) := pure ⟨cmd, tsid, tac, none⟩
      | exc := tactic.fail format!"request_parsing_error: cmd={cmd} data={exc}"
      end
    | "query_env" := match json.array args with
      | (json.array [json.of_int (int.of_nat tsid)]) := pure ⟨cmd, tsid, none, none⟩
      | exc := tactic.fail format!"request_parsing_error: cmd={cmd} data={exc}"
      end
    | "query_decl" := match json.array args with
      | (json.array [json.of_int (int.of_nat tsid), json.of_string name]) := pure ⟨cmd, tsid, none, name⟩
      | exc := tactic.fail format!"request_parsing_error: cmd={cmd} data={exc}"
      end
    | "exit_repl" := match json.array args with
      | json.array [] := pure ⟨cmd, none, none, none⟩
      | exc := tactic.fail format!"request_parsing_error: cmd={cmd} data={exc}"
      end
    | exc := tactic.fail format!"request_parsing_error: data={exc}"
    end
  | exc := tactic.fail format!"request_parsing_error: data={exc}"
  end
⟩

meta structure LeanREPLResponse : Type :=
(sid : option nat)
(tacticState : option string)  -- To be consistent with Lean 4's
(env_fingerprint: option string)
(environment: option (list string))
(declaration: option (list (string × string)))
(error: option string)

meta def LeanREPLResponse.to_json: LeanREPLResponse → json
| ⟨tsid, ts, ef, env, decl, err⟩ :=
    json.object [
      ⟨"sid", match tsid with
        | none := json.null
        | some tsid := json.of_int (int.of_nat tsid)
        end⟩,
      ⟨"tacticState", match ts with
        | none := json.null
        | some ts := json.of_string ts
        end⟩,
      ⟨"env_fingerprint", match ef with
        | none := json.null
        | some ef := json.of_string ef
        end⟩,
      ⟨"environment", match env with
        | none := json.null
        | some env := json.array $ env.map json.of_string
        end⟩,
      ⟨"declaration", match decl with
        | none := json.null
        | some decl := json.object $ decl.map (λ kv, (kv.fst, json.of_string kv.snd))
        end⟩,        
      ⟨"error", match err with
        | none := json.null
        | some err := json.of_string err
        end⟩
    ]

meta instance : has_to_format LeanREPLResponse :=
⟨has_to_format.to_format ∘ LeanREPLResponse.to_json⟩

meta structure parent : Type :=
(tsid : nat)
(tactic : string)

meta structure LeanREPLState : Type :=
(states : list (tactic_state × option parent))
(ready_to_exit: bool)

namespace LeanREPLState

meta def empty : LeanREPLState := ⟨[], false⟩

meta def insert_ts (σ : LeanREPLState) (tsid : nat) (ts : tactic_state) (p: option parent) (is_success : bool) : LeanREPLState := 
  let new_states := σ.states.append [(ts, p)] in
  ⟨new_states, σ.ready_to_exit⟩

meta def get_ts_parents (σ : LeanREPLState) (tsid : nat) : option (tactic_state × option parent) := 
  σ.states.nth tsid

meta def get_ts (σ : LeanREPLState) (tsid : nat) : option tactic_state := 
  option.map prod.fst $ σ.get_ts_parents tsid

meta def get_next_tsid (σ : LeanREPLState) : nat := σ.states.length

end LeanREPLState

@[reducible]
meta def LeanREPL := state_t LeanREPLState io

meta def parse_request (msg : string) : io LeanREPLRequest := do {
  match json.parse msg with
  | (some json_msg) := io.run_tactic'' $ has_from_json.from_json json_msg
  | none := io.fail' format!"[fatal] parse_failed: data={msg}"
  end
}

meta def record_ts (ts : tactic_state) (parent : option parent) (is_success : bool) : LeanREPL nat := do {
  σ ← get,
  let tsid := σ.get_next_tsid,
  modify $ λ σ, σ.insert_ts tsid ts parent is_success,
  pure tsid
}

meta def postprocess_tactic_state (ts : tactic_state) : tactic string := do
  -- Note: we do not postprocess here, because we assume that there are other
  -- data sources that use default `pp` settings.
  pure $ to_string (to_fmt ts)

/-- `has_local_constant e l` checks whether local constant `l` occurs in expression `e` -/
meta def has_local_constant (e l : expr) : bool :=
e.has_local_in $ mk_name_set.insert l.local_uniq_name

meta def abstract_all_locals (ctor : name → binder_info → expr → expr → expr) (e : expr) (ctx : list expr) : tactic expr := ctx.reverse.mfoldl (λ body v, do {
  let var_name := expr.local_pp_name v,
  var_type ← tactic.get_local_type var_name,
  pure $ ctor var_name binder_info.default var_type (body.abstract_local (local_uniq_name v))
}) e

meta def get_env_fingerprint (ts : tactic_state) : tactic string := do {
  env ← get_env,
  return $ to_string $ environment.fingerprint env
}

meta def finalize_proof (tsid : nat) (tac : string) (ts' : tactic_state) : LeanREPL LeanREPLResponse := do {
  σ ← get,
  -- Retrieve the tactic state at index 0 to extract the top-level goal metavariable.
  match σ.get_ts 0 with
  | none := do {
    let err := format! "unexpected_unknown_tsid_0",
    pure ⟨none, none, none, none, none, err.to_string⟩
  }
  | (some ts₀) := do {
    result ← (state_t.lift ∘ io.run_tactic'') $ do {
      -- Set to tactic state index 0 to retrieve the meta-variable for the top goal.
      tactic.write ts₀,
      [g] ← tactic.get_goals,
      tgt ← tactic.infer_type g,
      ctx ← local_context,

      tactic.write ts',
      pf ← tactic.get_assignment g >>= tactic.instantiate_mvars,
      
      tactic.write ts₀,
      pf ← abstract_all_locals expr.lam pf ctx,
      tgt ← abstract_all_locals expr.pi tgt ctx,

      tactic.write ts',
      tactic.capture' (validate_proof tgt pf)
    },
    match result with
    | (interaction_monad.result.success r s') := do {
      tsid ← record_ts ts' (some ⟨tsid, tac⟩) tt,
      ts_str ← state_t.lift $ io.run_tactic'' $ postprocess_tactic_state ts',
      ef ← state_t.lift $ io.run_tactic'' $ get_env_fingerprint ts',
      pure $ ⟨tsid, ts_str, ef, none, none, none⟩
    }
    | (interaction_monad.result.exception f p s') := do {
      let msg := (f.get_or_else (λ _, format.of_string "n/a")) (),
      let err := format! "proof_validation_failed: msg={msg}",
      pure ⟨none, none, none, none, none, err.to_string⟩
    }
    end
  }
  end
}

meta def handle_run_tac (tsid : nat) (tac : string) : LeanREPL LeanREPLResponse := do {
  σ ← get,
  match (σ.get_ts tsid) with
  -- Received an unknown search id, return an error.
  | none := do {
    let err := format!"unknown_id: tactic_state_id={tsid}",
    pure ⟨none, none, none, none, none, err.to_string⟩
  }
  -- The tactic state was retrieved from the state.
  | (some ts) := do {
    -- Set the tactic state and try to apply the tactic.
    result_with_string ← state_t.lift $ io.run_tactic'' $ do {
      tactic.write ts,
      get_tac_and_capture_result tac $TACTIC_TIMEOUT <|> do {
          let msg : format := format!"parse_itactic failed on `{tac}`",
          interaction_monad.mk_exception msg none <$> tactic.read
      }
    },
    match result_with_string with
    -- The tactic application was successful.
    | interaction_monad.result.success _ ts' := do {
        n ← (state_t.lift ∘ io.run_tactic'') $ do {
          tactic.write ts',
          tactic.num_goals
        },
        -- monad_lift $ io.run_tactic'' $ tactic.trace format! "REMAINING SUBGOALS: {n}",
        match n with
        -- There is no more subgoals, check that the produced proof is valid.
        | 0 := do {
          finalize_proof tsid tac ts'
        }
        -- There are remaining subgoals, return the updated tactic state.
        | n := do {
          tsid ← record_ts ts' (some ⟨tsid, tac⟩) ff,
          ts_str ← state_t.lift $ io.run_tactic'' $ postprocess_tactic_state ts',
          ef ← state_t.lift $ io.run_tactic'' $ get_env_fingerprint ts',
          pure $ ⟨tsid, ts_str, ef, none, none, none⟩
        }
        end
      }
    -- The tactic application failed, potentially return an error with the failure message.
    | interaction_monad.result.exception fn pos ts' := do {
        -- Some tactics such as linarith fail but result in a tactic state with no goals. Check if
        -- that's the case and finalize the proof, otherwise error.
        n ← state_t.lift $ io.run_tactic'' $ do {
          tactic.write ts',
          tactic.num_goals
        },
        -- monad_lift $ io.run_tactic'' $ tactic.trace format! "REMAINING SUBGOALS: {n}",
        match n with
        -- There is no more subgoals, check that the produced proof is valid.
        | 0 := do {
          finalize_proof tsid tac ts'
        }
        -- There are remaining subgoals, return the error.
        | _ := do {
          state_t.lift $ do {
            let msg := (fn.get_or_else (λ _, format.of_string "n/a")) (),
            -- let io.run_tactic'' $ trace_state,
            ts_str ← io.run_tactic'' $ postprocess_tactic_state ts',
            let err := format! "gen_tac_and_capture_res_failed: pos={pos} msg=\"{msg}\" tactic_state=\"{ts_str}\"",
            pure ⟨none, none, none, none, none, err.to_string⟩
          }
        }
        end
      }
    end
  }
  end
}

meta def get_declarations (env : environment) : list declaration := 
  env.fold [] (λ d decls, d::decls)

meta def handle_query_env (tsid : nat) : LeanREPL LeanREPLResponse := do {
  σ ← get,
  match (σ.get_ts tsid) with
  -- Received an unknown search id, return an error.
  | none := do {
    let err := format!"unknown_id: tactic_state_id={tsid}",
    pure ⟨none, none, none, none, none, err.to_string⟩
  }
  -- The tactic state was retrieved from the state.
  | (some ts) := do {
    let decls := get_declarations $ tactic_state.env ts,
    let env := decls.map (λ d,  to_string d.to_name),
    pure ⟨none, none, none, some env, none, none⟩
  }
  end
}

meta def is_numeral (s : string) : bool :=
  s.fold tt (λ r c, r && c.is_digit)

meta def string_to_name (s : string) : name :=
  let fields := string.split (λ c, c = '.') s in
  fields.foldl (λ n s, if is_numeral s then mk_num_name n s.to_nat else mk_str_name n s) name.anonymous 

meta def handle_query_decl (tsid : nat) (s : string) : LeanREPL LeanREPLResponse := do {
  σ ← get,
  match (σ.get_ts tsid) with
  -- Received an unknown search id, return an error.
  | none := do {
    let err := format!"unknown_id: tactic_state_id={tsid}",
    pure ⟨none, none, none, none, none, err.to_string⟩
  }
  -- The tactic state was retrieved from the state.
  | (some ts) := do {
    let env := tactic_state.env ts,
    let name := string_to_name s,
    match env.get $ name with 
    | exceptional.success decl := do {
      srt ← state_t.lift $ io.run_tactic'' $ tactic.infer_type decl.type,
      type_fmt ← state_t.lift $ io.run_tactic''$ tactic_format_expr decl.type,
      let type := to_string type_fmt,
      let is_inductive := to_string $ env.is_inductive name,
      let is_constructor := to_string $ env.is_constructor name,
      let is_recursor := to_string $ env.is_recursor name,
      let is_recursive := to_string $ env.is_recursive name,
      let inductive_type_of := to_string $ env.inductive_type_of name,
      let constructors_of := to_string $ env.constructors_of name,
      let recursor_of := to_string $ env.recursor_of name,
      let inductive_num_params := to_string $ env.inductive_num_params name,
      let inductive_num_indices := to_string $ env.inductive_num_indices name,
      let inductive_dep_elim := to_string $ env.inductive_dep_elim name,
      let is_ginductive := to_string $ env.is_ginductive name,
      let relation_info := to_string $ env.relation_info name,
      let refl_for := to_string $ env.refl_for name,
      let symm_for := to_string $ env.symm_for name,
      let trans_for := to_string $ env.trans_for name,
      let decl_olean := to_string $ env.decl_olean name,
      let structure_fields := to_string $ env.structure_fields name,
      let eqn_lemmas := to_string $ env.get_eqn_lemmas_for name,
      let ext_eqn_lemmas := to_string $ env.get_ext_eqn_lemmas_for name,
      let in_current_file := to_string $ env.in_current_file name,
      let is_definition := to_string $ env.is_definition name,
      pure ⟨none, none, none, none, some [
        ("type", type), 
        ("sort", to_string srt),
        ("is_inductive", is_inductive), 
        ("is_constructor", is_constructor), 
        ("is_recursor", is_recursor), 
        ("is_recursive", is_recursive), 
        ("inductive_type_of", inductive_type_of), 
        ("constructors_of", constructors_of), 
        ("recursor_of", recursor_of), 
        ("inductive_num_params", inductive_num_params), 
        ("inductive_num_indices", inductive_num_indices), 
        ("inductive_dep_elim", inductive_dep_elim), 
        ("is_ginductive", is_ginductive), 
        ("relation_info", relation_info), 
        ("refl_for", refl_for), 
        ("symm_for", symm_for), 
        ("trans_for", trans_for), 
        ("decl_olean", decl_olean), 
        ("structure_fields", structure_fields), 
        ("eqn_lemmas", eqn_lemmas), 
        ("ext_eqn_lemmas", ext_eqn_lemmas), 
        ("in_current_file", in_current_file), 
        ("is_definition", is_definition)
      ], none⟩ 
    }
    | _ := do {
      let err := format!"unknown_declaration: name={name}",
      pure ⟨none, none, none, none, none, err.to_string⟩
    }
    end
  }
  end
}

meta def handle_exit_repl : LeanREPL LeanREPLResponse := do {
  σ ← get,
  modify $ λ σ, ⟨σ.states, tt⟩,
  pure ⟨none, none, none, none, none, none⟩ 
}

meta def handle_request (req : LeanREPLRequest) : LeanREPL LeanREPLResponse := 
match req.cmd, req.tsid, req.tac, req.name with
  | "run_tac", (some tsid), (some tac), none := handle_run_tac tsid tac
  | "query_env", (some tsid), none, none := handle_query_env tsid
  | "query_decl", (some tsid), none, (some name) := handle_query_decl tsid name
  | "exit_repl", none, none, none := handle_exit_repl
  | cmd, tsid, tac, name := state_t.lift $ io.fail' format!"[fatal] invalid_command: cmd={cmd}, tactic_state_id={tsid}, tactic={tac}, name={name}"
end

meta def loop : LeanREPL unit := do {
  req ← (state_t.lift $ io.get_line >>= parse_request),
  res ← handle_request req,
  state_t.lift $ io.put_str_ln' format!"REPL> {(json.unparse ∘ LeanREPLResponse.to_json) res}"
}

meta def LeanREPL.iterate (x : LeanREPL unit) : LeanREPL unit := do {
  σ₀ ← get,
  state_t.lift $ io.iterate σ₀ $ λ σ, do {
    (_, σ') ← x.run σ,
    match σ'.ready_to_exit with
    | tt := return none
    | ff := return (some σ')
    end
  },
  pure ()
}

meta def is_theorem : tactic bool := target >>= is_prop

meta def initialize : tactic tactic_state := do {
  is_thm ← is_theorem,
  match is_thm with
  -- The declaration is not a theorem, return an error.
  | ff := tactic.fail format!"[fatal] not_a_theorem"
  -- The declaration is a theorem, generate a new tactic state.
  | tt := do {
    ts ← tactic.read,
    ts_str ← postprocess_tactic_state ts,
    ef ← get_env_fingerprint ts,
    let res : LeanREPLResponse := ⟨some 0, ts_str, ef, none, none, none⟩,
    unsafe_run_io $ io.put_str_ln' format!"REPL> {(json.unparse ∘ LeanREPLResponse.to_json) res}",
    pure ts
  }
  end
}

private meta def enable_full_names : tactic unit := do {
  set_bool_option `pp.full_names true
}

private meta def with_full_names {α} (tac : tactic α) : tactic α :=
tactic.save_options $ enable_full_names *> tac

meta def repl : tactic unit := do {
  ts ← initialize,
  let σ₀ : LeanREPLState := ⟨[(ts, none)], false⟩,
  with_full_names $ unsafe_run_io $ state_t.run loop.iterate σ₀ $> ()
}

end lean_dojo