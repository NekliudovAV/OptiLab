# Functions for Equation Analysis:
# Проводит анализ уравнения и если есть небаланс по исходным данным, возвращает величину небаланса
from pyomo.repn.standard_repn import generate_standard_repn
def presolve_eq(Eq):
    out=generate_standard_repn(Eq.body)
    value=out.constant
    linearVars=out.linear_vars
    QuadricVars=out.quadratic_vars
    # Линейная составляющая
    i=0
    val=0
    Koefs=out.linear_coefs
    for v in linearVars:
        if v.value is None:
            val=np.inf
            print(v.name,val,Koefs[i])
        else:
            val=v.value
        value=value+val*Koefs[i]
        i=i+1;
        
    # Квадратичная составляющая
    i=0
    Koefs=out.quadratic_coefs
    val=[0,0]
    for v in QuadricVars:
        if v[0].value is None:
            val[0]=np.inf
            print(v[0].name,v[0].value,v[1].name,v[1].value,Koefs[i])
        else:
            val[0]=v[0].value
            
        if v[1].value is None:
            val[1]=np.inf
            print(v[0].name,v[0].value,v[1].name,v[1].value,Koefs[i])
        else:
            val[1]=v[1].value
            
        
        value=value+val[0]*val[1]*Koefs[i]
        i=i+1;
    
    #print(value)
    return value

# Проверяет каждое уравнение на предмет небаланса
def analise(mdh):
    sortOrder = SortComponents.unsorted
    correct_eq=[]
    delta=[]
    for constraint_data in mdh.component_data_objects(
                            Constraint,
                            active=True,
                            sort=sortOrder,
                            descend_into=False):
        print(constraint_data.name)
        value=presolve_eq(constraint_data)
        print(constraint_data.name,' - небаланс: ',value)
        if abs(value)>0 and not np.isinf(value) :
            correct_eq.append(constraint_data)
            delta.append(value)
    return correct_eq, delta

# Функция коррекируем уравнение Eq: добавляет в него DeltaPlus и DeltaMinus Ex=Ex+DeltaPlus[j]-DeltaMinus[j]
# * mdh.DeltaPlus и mdh.DeltaMinus должны быть уже созданы в модели
def gen_Expression(mdh,Eq,j):
    out=generate_standard_repn(Eq.body)
    value=out.constant
    linearVars=out.linear_vars
    QuadricVars=out.quadratic_vars
    # Линейная составляющая
    i=0
    val=0
    Koefs=out.linear_coefs
    for v in linearVars:
        value=value+v*Koefs[i]
        i=i+1;
    # Квадратичная составляющая
    i=0
    Koefs=out.quadratic_coefs
    for v in QuadricVars:
        value=value+v[0]*v[1]*Koefs[i]
        i=i+1;
    Expr=value+mdh.DeltaPlus[j]-mdh.DeltaMinus[j]
    #print(Expr)
    mdh.Constr.add(Expr==0)     
    return Expr
