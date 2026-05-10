// Rearranges the elements in an array 'a' of natural numbers,
// so that all odd numbers appear before all even numbers.
method partitionOddEven(a: array<nat>) 
  modifies a
  ensures multiset(a[..]) == multiset(old(a[..]))
  ensures ! exists i, j :: 0 <= i < j < a.Length && even(a[i]) && odd(a[j])  
{
<<<BODY>>>
}
 
predicate  odd(n: nat) { n % 2 == 1 }
predicate  even(n: nat) { n % 2 == 0 }

method testPartitionOddEven() {
    var a: array<nat> := new [] [1, 2, 3];
    assert a[..] == [1, 2, 3];
    partitionOddEven(a);
    assert a[..] == [1, 3, 2] || a[..] == [3, 1, 2];
}

