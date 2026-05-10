"""Reference implementations (verified against specs) for --golden smoke runs without an LLM."""

BODIES: dict[str, str] = {
    "binary_search": (
        "  var lo := 0;\n"
        "  var hi := a.Length;\n"
        "  while lo < hi\n"
        "    invariant 0 <= lo <= hi <= a.Length\n"
        "    invariant forall k :: 0 <= k < lo ==> a[k] < x\n"
        "    invariant forall k :: hi <= k < a.Length ==> a[k] > x\n"
        "    decreases hi - lo\n"
        "  {\n"
        "    var mid := lo + (hi - lo) / 2;\n"
        "    if a[mid] < x {\n"
        "      lo := mid + 1;\n"
        "    } else if x < a[mid] {\n"
        "      hi := mid;\n"
        "    } else {\n"
        "      index := mid;\n"
        "      return;\n"
        "    }\n"
        "  }\n"
        "  index := -1;\n"
    ),
    "max_index": (
        "  i := 0;\n"
        "  var j := 1;\n"
        "  while j < a.Length\n"
        "    invariant 0 <= i < a.Length\n"
        "    invariant 1 <= j <= a.Length\n"
        "    invariant forall k :: 0 <= k < j ==> a[k] <= a[i]\n"
        "    decreases a.Length - j\n"
        "  {\n"
        "    if a[j] > a[i] {\n"
        "      i := j;\n"
        "    }\n"
        "    j := j + 1;\n"
        "  }\n"
    ),
    "reverse_seq": (
        "  if |s| == 0 {\n"
        "    t := [];\n"
        "  } else {\n"
        "    var r := ReverseInts(s[..|s| - 1]);\n"
        "    t := [s[|s| - 1]] + r;\n"
        "  }\n"
    ),
}
