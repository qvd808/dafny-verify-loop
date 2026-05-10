"""Few-shot examples for guiding limited LLMs — structurally matched to problem types."""

from __future__ import annotations

# Each example: (spec_description, template, correct_body, problem_type)
# problem_type is used to select relevant examples

Example = tuple[str, str, str, str]


def _make_binary_search_example() -> Example:
    desc = "Binary search on sorted array: find target index or return -1."
    template = """predicate Sorted(a: array<int>)
  reads a
{
  forall i, j :: 0 <= i < j < a.Length ==> a[i] <= a[j]
}

method BinarySearch(a: array<int>, x: int) returns (index: int)
  requires Sorted(a)
  ensures index >= 0 ==> index < a.Length && a[index] == x
  ensures index < 0 ==> forall k :: 0 <= k < a.Length ==> a[k] != x
{
<<<BODY>>>
}"""
    body = """  var lo := 0;
  var hi := a.Length;
  while lo < hi
    invariant 0 <= lo <= hi <= a.Length
    invariant forall k :: 0 <= k < lo ==> a[k] < x
    invariant forall k :: hi <= k < a.Length ==> a[k] > x
    decreases hi - lo
  {
    var mid := lo + (hi - lo) / 2;
    if a[mid] < x {
      lo := mid + 1;
    } else if x < a[mid] {
      hi := mid;
    } else {
      index := mid;
      return;
    }
  }
  index := -1;"""
    return (desc, template, body, "binary_search")


def _make_max_index_example() -> Example:
    desc = "Find index of maximum element in non-empty array."
    template = """method FindMaxIndex(a: array<int>) returns (i: int)
  requires a.Length > 0
  ensures 0 <= i < a.Length
  ensures forall k :: 0 <= k < a.Length ==> a[k] <= a[i]
{
<<<BODY>>>
}"""
    body = """  i := 0;
  var j := 1;
  while j < a.Length
    invariant 0 <= i < a.Length
    invariant 1 <= j <= a.Length
    invariant forall k :: 0 <= k < j ==> a[k] <= a[i]
    decreases a.Length - j
  {
    if a[j] > a[i] {
      i := j;
    }
    j := j + 1;
  }"""
    return (desc, template, body, "linear_scan")


def _make_reverse_example() -> Example:
    desc = "Reverse a sequence recursively."
    template = """method ReverseSeq(s: seq<int>) returns (t: seq<int>)
  ensures |t| == |s|
  ensures forall i :: 0 <= i < |s| ==> t[i] == s[|s| - 1 - i]
{
<<<BODY>>>
}"""
    body = """  if |s| == 0 {
    t := [];
  } else {
    var r := ReverseSeq(s[..|s| - 1]);
    t := [s[|s| - 1]] + r;
  }"""
    return (desc, template, body, "recursive")


def _make_sum_example() -> Example:
    desc = "Compute sum of array elements iteratively."
    template = """method ArraySum(a: array<int>) returns (sum: int)
  ensures sum == SumRange(a, 0, a.Length)

  function SumRange(a: array<int>, lo: int, hi: int): int
    reads a
    requires 0 <= lo <= hi <= a.Length
    decreases hi - lo
  {
    if lo == hi then 0 else a[lo] + SumRange(a, lo + 1, hi)
  }
{
<<<BODY>>>
}"""
    body = """  sum := 0;
  var j := 0;
  while j < a.Length
    invariant 0 <= j <= a.Length
    invariant sum == SumRange(a, 0, j)
    decreases a.Length - j
  {
    sum := sum + a[j];
    j := j + 1;
  }"""
    return (desc, template, body, "linear_scan")


def _make_two_sum_example() -> Example:
    desc = "Find two distinct indices whose values sum to target, or return (-1, -1)."
    template = """method TwoSum(a: array<int>, target: int) returns (i: int, j: int)
  ensures (i >= 0 && j >= 0) ==> i < a.Length && j < a.Length && i != j && a[i] + a[j] == target
  ensures (i < 0 || j < 0) ==> forall p, q :: 0 <= p < q < a.Length ==> a[p] + a[q] != target
{
<<<BODY>>>
}"""
    body = """  i := -1;
  j := -1;
  var p := 0;
  while p < a.Length
    invariant 0 <= p <= a.Length
    invariant i < 0 || j < 0 ==> forall u, v :: 0 <= u < v < p ==> a[u] + a[v] != target
    decreases a.Length - p
  {
    var q := p + 1;
    while q < a.Length
      invariant p + 1 <= q <= a.Length
      invariant i < 0 || j < 0 ==> forall u :: 0 <= u < p ==> a[u] + a[q-1] != target
      decreases a.Length - q
    {
      if a[p] + a[q] == target {
        i := p;
        j := q;
        return;
      }
      q := q + 1;
    }
    p := p + 1;
  }"""
    return (desc, template, body, "nested_loop")


ALL_EXAMPLES: list[Example] = [
    _make_binary_search_example(),
    _make_max_index_example(),
    _make_reverse_example(),
    _make_sum_example(),
    _make_two_sum_example(),
]


def get_examples_by_type(problem_type: str) -> list[Example]:
    """Return examples matching the given problem type."""
    return [ex for ex in ALL_EXAMPLES if ex[3] == problem_type]


def get_all_examples() -> list[Example]:
    return list(ALL_EXAMPLES)


def format_example(example: Example, include_body: bool = True) -> str:
    """Format a single example for inclusion in prompts."""
    desc, template, body, ptype = example
    parts = [f"// Example ({ptype}): {desc}", "", template.replace("<<<BODY>>>", body)]
    return "\n".join(parts)


def format_examples_for_prompt(examples: list[Example], max_examples: int = 2) -> str:
    """Format examples for inclusion in an LLM prompt."""
    selected = examples[:max_examples]
    parts = ["// ===== SOLVED EXAMPLES FOR REFERENCE =====", ""]
    for ex in selected:
        parts.append(format_example(ex))
        parts.append("")
    parts.append("// ===== END EXAMPLES =====")
    return "\n".join(parts)
