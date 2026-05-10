method Getmini(a:array<int>) returns(mini:nat) 
requires a.Length > 0
ensures 0 <= mini < a.Length // mini is an index of a
ensures forall x :: 0 <= x < a.Length ==> a[mini] <= a[x] // a[mini] is the minimum value
ensures forall x :: 0 <= x < mini ==> a[mini] < a[x] // a[mini] is the first min
{
<<<BODY>>>
}

/*
method check() {
    var data := new int[][9,5,42,5,5]; // minimum 5 first at index 1
var mini := Getmini(data);
//print mini;
assert mini==1;

}
*/

