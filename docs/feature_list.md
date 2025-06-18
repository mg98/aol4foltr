# Feature List

Each query-url pair is represented by a 103-dimensional vector, see [code](./ltr/types.py).

<table>
<thead>
<tr>
<th>ID</th>
<th>Description</th>
<th>Stream</th>
</tr>
</thead>
<tbody>
<tr><td>1</td><td rowspan="3">BM25</td><td>title</td></tr>
<tr><td>2</td><td>body</td></tr>
<tr><td>3</td><td>url</td></tr>
<tr><td>4</td><td rowspan="3">Min of term frequency (TF)</td><td>title</td></tr>
<tr><td>5</td><td>body</td></tr>
<tr><td>6</td><td>url</td></tr>
<tr><td>7</td><td rowspan="3">Max of term frequency (TF)</td><td>title</td></tr>
<tr><td>8</td><td>body</td></tr>
<tr><td>9</td><td>url</td></tr>
<tr><td>10</td><td rowspan="3">Sum of term frequency (TF)</td><td>title</td></tr>
<tr><td>11</td><td>body</td></tr>
<tr><td>12</td><td>url</td></tr>
<tr><td>13</td><td rowspan="3">Mean of term frequency (TF)</td><td>title</td></tr>
<tr><td>14</td><td>body</td></tr>
<tr><td>15</td><td>url</td></tr>
<tr><td>16</td><td rowspan="3">Variance of term frequency (TF)</td><td>title</td></tr>
<tr><td>17</td><td>body</td></tr>
<tr><td>18</td><td>url</td></tr>
<tr><td>19</td><td rowspan="3">Min of inverse document frequency (IDF)</td><td>title</td></tr>
<tr><td>20</td><td>body</td></tr>
<tr><td>21</td><td>url</td></tr>
<tr><td>22</td><td rowspan="3">Max of inverse document frequency (IDF)</td><td>title</td></tr>
<tr><td>23</td><td>body</td></tr>
<tr><td>24</td><td>url</td></tr>
<tr><td>25</td><td rowspan="3">Sum of inverse document frequency (IDF)</td><td>title</td></tr>
<tr><td>26</td><td>body</td></tr>
<tr><td>27</td><td>url</td></tr>
<tr><td>28</td><td rowspan="3">Mean of inverse document frequency (IDF)</td><td>title</td></tr>
<tr><td>29</td><td>body</td></tr>
<tr><td>30</td><td>url</td></tr>
<tr><td>31</td><td rowspan="3">Variance of inverse document frequency (IDF)</td><td>title</td></tr>
<tr><td>32</td><td>body</td></tr>
<tr><td>33</td><td>url</td></tr>
<tr><td>34</td><td rowspan="3">Min of TF*IDF</td><td>title</td></tr>
<tr><td>35</td><td>body</td></tr>
<tr><td>36</td><td>url</td></tr>
<tr><td>37</td><td rowspan="3">Max of TF*IDF</td><td>title</td></tr>
<tr><td>38</td><td>body</td></tr>
<tr><td>39</td><td>url</td></tr>
<tr><td>40</td><td rowspan="3">Sum of TF*IDF</td><td>title</td></tr>
<tr><td>41</td><td>body</td></tr>
<tr><td>42</td><td>url</td></tr>
<tr><td>43</td><td rowspan="3">Mean of TF*IDF</td><td>title</td></tr>
<tr><td>44</td><td>body</td></tr>
<tr><td>45</td><td>url</td></tr>
<tr><td>46</td><td rowspan="3">Variance of TF*IDF</td><td>title</td></tr>
<tr><td>47</td><td>body</td></tr>
<tr><td>48</td><td>url</td></tr>
<tr><td>49</td><td rowspan="3">Min of stream length</td><td>title</td></tr>
<tr><td>50</td><td>body</td></tr>
<tr><td>51</td><td>url</td></tr>
<tr><td>52</td><td rowspan="3">Max of stream length</td><td>title</td></tr>
<tr><td>53</td><td>body</td></tr>
<tr><td>54</td><td>url</td></tr>
<tr><td>55</td><td rowspan="3">Sum of stream length</td><td>title</td></tr>
<tr><td>56</td><td>body</td></tr>
<tr><td>57</td><td>url</td></tr>
<tr><td>58</td><td rowspan="3">Mean of stream length</td><td>title</td></tr>
<tr><td>59</td><td>body</td></tr>
<tr><td>60</td><td>url</td></tr>
<tr><td>61</td><td rowspan="3">Variance of stream length</td><td>title</td></tr>
<tr><td>62</td><td>body</td></tr>
<tr><td>63</td><td>url</td></tr>
<tr><td>64</td><td rowspan="3">Min of stream length normalized TF</td><td>title</td></tr>
<tr><td>65</td><td>body</td></tr>
<tr><td>66</td><td>url</td></tr>
<tr><td>67</td><td rowspan="3">Max of stream length normalized TF</td><td>title</td></tr>
<tr><td>68</td><td>body</td></tr>
<tr><td>69</td><td>url</td></tr>
<tr><td>70</td><td rowspan="3">Sum of stream length normalized TF</td><td>title</td></tr>
<tr><td>71</td><td>body</td></tr>
<tr><td>72</td><td>url</td></tr>
<tr><td>73</td><td rowspan="3">Mean of stream length normalized TF</td><td>title</td></tr>
<tr><td>74</td><td>body</td></tr>
<tr><td>75</td><td>url</td></tr>
<tr><td>76</td><td rowspan="3">Variance of stream length normalized TF</td><td>title</td></tr>
<tr><td>77</td><td>body</td></tr>
<tr><td>78</td><td>url</td></tr>
<tr><td>79</td><td rowspan="3">Cosine similarity</td><td>title</td></tr>
<tr><td>80</td><td>body</td></tr>
<tr><td>81</td><td>url</td></tr>
<tr><td>82</td><td rowspan="3">Covered query term number</td><td>title</td></tr>
<tr><td>83</td><td>body</td></tr>
<tr><td>84</td><td>url</td></tr>
<tr><td>85</td><td rowspan="3">Covered query term ratio</td><td>title</td></tr>
<tr><td>86</td><td>body</td></tr>
<tr><td>87</td><td>url</td></tr>
<tr><td>88</td><td rowspan="3">Character length</td><td>title</td></tr>
<tr><td>89</td><td>body</td></tr>
<tr><td>90</td><td>url</td></tr>
<tr><td>91</td><td rowspan="3">Term length</td><td>title</td></tr>
<tr><td>92</td><td>body</td></tr>
<tr><td>93</td><td>url</td></tr>
<tr><td>94</td><td rowspan="3">Total query terms</td><td>title</td></tr>
<tr><td>95</td><td>body</td></tr>
<tr><td>96</td><td>url</td></tr>
<tr><td>97</td><td rowspan="3">Exact match (bool)</td><td>title</td></tr>
<tr><td>98</td><td>body</td></tr>
<tr><td>99</td><td>url</td></tr>
<tr><td>100</td><td rowspan="3">Match ratio</td><td>title</td></tr>
<tr><td>101</td><td>body</td></tr>
<tr><td>102</td><td>url</td></tr>
<tr><td>103</td><td>Number of slahes</td><td>url</td></tr>
</tbody>
</table>
