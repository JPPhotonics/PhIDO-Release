% ============================================================================
% PhIDO Architecture Rules — Clingo ASP topology validator
%
% Input facts (emitted by clingo_validator.py):
%   component(ID, Type, SubType).
%   connection(FromID, ToID).
%   architecture(Atom).           % e.g. architecture(mzi).
%   n_value(N).                   % scaling parameter
%   requirement(ReqID).
%   requirement_trace(ReqID, SatisfactionType).
%
% Output: error/2..3 atoms for each detected topology violation.
% ============================================================================

% ---------------------------------------------------------------------------
% Helper: count components by type
% ---------------------------------------------------------------------------
component_count(Type, N) :- N = #count { ID : component(ID, Type, _) }, component(_, Type, _).

% ---------------------------------------------------------------------------
% MZI rules
% ---------------------------------------------------------------------------
% An MZI architecture requires exactly 1 splitter and 1 combiner,
% and the splitter must be connected to the combiner (via any path).

splitter_count(N) :- N = #count { ID : component(ID, splitter, _) }.
combiner_count(N) :- N = #count { ID : component(ID, combiner, _) }.

% Transitive connectivity
connected(X, Y) :- connection(X, Y).
connected(X, Y) :- connected(X, Z), connection(Z, Y).

has_splitter_combiner_path :-
    component(S, splitter, _),
    component(C, combiner, _),
    connected(S, C).

error(mzi_wrong_splitter_count, N) :-
    architecture(mzi),
    splitter_count(N),
    N != 1.

error(mzi_wrong_combiner_count, N) :-
    architecture(mzi),
    combiner_count(N),
    N != 1.

error(mzi_disconnected) :-
    architecture(mzi),
    splitter_count(1),
    combiner_count(1),
    not has_splitter_combiner_path.

% ---------------------------------------------------------------------------
% Splitter tree rules
% ---------------------------------------------------------------------------
% A 1xN splitter tree requires N-1 splitters, and N must be a power of 2.

is_power_of_2(2). is_power_of_2(4). is_power_of_2(8). is_power_of_2(16).
is_power_of_2(32). is_power_of_2(64). is_power_of_2(128).

error(tree_not_power_of_2, N) :-
    architecture(splitter_tree),
    n_value(N),
    not is_power_of_2(N).

tree_expected(N, N-1) :- n_value(N), architecture(splitter_tree).

error(tree_wrong_splitters, N, Actual) :-
    architecture(splitter_tree),
    n_value(N),
    component_count(splitter, Actual),
    tree_expected(N, Expected),
    Actual != Expected.

% ---------------------------------------------------------------------------
% Benes network rules
% ---------------------------------------------------------------------------
% A Benes NxN network requires (N/2)*log2(N)*2 - N/2 switches.
% Precomputed for common sizes:
%   4x4: 6 switches    8x8: 20 switches

benes_expected(4, 6).
benes_expected(8, 20).

% Count switching units (MZMs or coupler-based switches)
switch_count(N) :- N = #count { ID : component(ID, mzm, _) }, component(_, mzm, _).
switch_count(0) :- not component(_, mzm, _).

error(benes_wrong_switch_count, N, Actual) :-
    architecture(benes),
    n_value(N),
    benes_expected(N, Expected),
    switch_count(Actual),
    Actual != Expected.

% ---------------------------------------------------------------------------
% Clements mesh rules
% ---------------------------------------------------------------------------
% An NxN Clements mesh requires N*(N-1)/2 MZI blocks.
% Precomputed: 4x4: 6   8x8: 28

clements_expected(4, 6).
clements_expected(8, 28).

error(clements_wrong_mzi_count, N, Actual) :-
    architecture(clements),
    n_value(N),
    clements_expected(N, Expected),
    switch_count(Actual),
    Actual != Expected.

% ---------------------------------------------------------------------------
% Reck mesh rules
% ---------------------------------------------------------------------------
% An NxN Reck (triangular) mesh also requires N*(N-1)/2 MZI blocks.
% Same counts as Clements for same N.

reck_expected(4, 6).
reck_expected(8, 28).

error(reck_wrong_mzi_count, N, Actual) :-
    architecture(reck),
    n_value(N),
    reck_expected(N, Expected),
    switch_count(Actual),
    Actual != Expected.

% ---------------------------------------------------------------------------
% WDM demux rules
% ---------------------------------------------------------------------------
% An N-channel WDM demux needs at least N ring resonators.

ring_count(N) :- N = #count { ID : component(ID, ring_resonator, _) }, component(_, ring_resonator, _).
ring_count(0) :- not component(_, ring_resonator, _).

error(wdm_insufficient_rings, N, Actual) :-
    architecture(wdm_demux),
    n_value(N),
    ring_count(Actual),
    Actual < N.

% ---------------------------------------------------------------------------
% QPSK modulator rules
% ---------------------------------------------------------------------------
% A QPSK modulator requires 1 splitter, 1 combiner, 2 MZMs, and a phase shifter.

mzm_count(N) :- N = #count { ID : component(ID, mzm, _) }, component(_, mzm, _).
mzm_count(0) :- not component(_, mzm, _).

phase_shifter_count(N) :- N = #count { ID : component(ID, phase_shifter, _) }, component(_, phase_shifter, _).
phase_shifter_count(0) :- not component(_, phase_shifter, _).

error(qpsk_wrong_splitter_count, N) :-
    architecture(qpsk),
    splitter_count(N),
    N != 1.

error(qpsk_wrong_combiner_count, N) :-
    architecture(qpsk),
    combiner_count(N),
    N != 1.

error(qpsk_wrong_mzm_count, N) :-
    architecture(qpsk),
    mzm_count(N),
    N != 2.

error(qpsk_missing_phase_shifter) :-
    architecture(qpsk),
    phase_shifter_count(0).

% ---------------------------------------------------------------------------
% Requirement traceability rules
% ---------------------------------------------------------------------------
% Every requirement must have a trace with satisfaction type other than "unaddressed".

error(unsatisfied_requirement, R) :-
    requirement(R),
    not requirement_trace(R, _).

error(unaddressed_requirement, R) :-
    requirement_trace(R, unaddressed).

% ---------------------------------------------------------------------------
% Show directives — only error atoms are projected into the answer set
% ---------------------------------------------------------------------------
#show error/1.
#show error/2.
#show error/3.
