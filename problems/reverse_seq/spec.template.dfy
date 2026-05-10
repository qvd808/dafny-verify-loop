// Spec S: reverse a seq<int> (BODY filled by generator).

method ReverseInts(s: seq<int>) returns (t: seq<int>)
  ensures |t| == |s|
  ensures forall i :: 0 <= i < |s| ==> t[i] == s[|s| - 1 - i]
{
<<<BODY>>>
}
