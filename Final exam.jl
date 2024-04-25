#Final exam

cd("C://Users//Mexbol//Desktop//maestria//Computational_Economics//FINAL")
using Pkg
Pkg.activate()
using Distributed
using Statistics
using DataFrames, CSV
using Plots
using Missings
addprocs(7)

@everywhere begin
  using Random
  using Combinatorics
  using LinearAlgebra
  using JuMP
end

##Machine Learning
using Flux


#read calibration (train) and competition (test) data, just combine them to create
#the new variables in an easier way, then we will split them again using their indices
calibration_data=CSV.read("All estimation raw data.csv", DataFrame)
calibration_data = calibration_data[:,3:end]
comp_data=CSV.read("raw-comp-set-data-Track-2.csv", DataFrame)
raw_data = vcat(calibration_data, comp_data)

#drop some variables we don't need like the subject ID and RT 
raw_data = select(raw_data, Not([:SubjID, :RT]))

rename!(raw_data, :B => :choice)
raw_data = select(raw_data, :choice, Not(:choice))
#First we will clean the variables to help with the prediction
# Location
raw_data.Location .= ifelse.(raw_data.Location .== "Rehovot", 1,
                            ifelse.(raw_data.Location .== "Technion", 2, missing))

# Gender
raw_data.Gender .= ifelse.(raw_data.Gender .== "F", 1,
                           ifelse.(raw_data.Gender .== "M", 2, missing))

# Condition
raw_data.Condition .= ifelse.(raw_data.Condition .== "ByFB", 1,
                              ifelse.(raw_data.Condition .== "ByProb", 2, missing))

# LotShapeA In this case we have some categorical variables, 
#if the Lottery is Left-skewed, the ouputs should be worst than a Right-skewed lottery
#So we will keep that order
raw_data.LotShapeA .= ifelse.(raw_data.LotShapeA .== "-", 1,
                              ifelse.(raw_data.LotShapeA .== "L-skew", 2,
                                      ifelse.(raw_data.LotShapeA .== "Symm", 3,
                                              ifelse.(raw_data.LotShapeA .== "R-skew", 4, missing))))

# LotShapeB
raw_data.LotShapeB .= ifelse.(raw_data.LotShapeB .== "-", 1,
                              ifelse.(raw_data.LotShapeB .== "L-skew", 2,
                                      ifelse.(raw_data.LotShapeB .== "Symm", 3,
                                              ifelse.(raw_data.LotShapeB .== "R-skew", 4, missing))))

# Button
raw_data.Button .= ifelse.(raw_data.Button .== "R", 1,
ifelse.(raw_data.Button .== "L", 2, missing))

X1=raw_data

@everywhere model="RUM"   # put "LA", "RCG", or "RUM"

## Common parameters
dYm=2                               # Number of varying options in menus
model=="RUM" ? dYu=2 : dYu=2        #
Menus=collect(powerset(vec(1:dYm))) # Menus

using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using Parameters: @with_kw

@with_kw mutable struct Args
    lr::Float64 = 0.5
    repeat::Int = 110
end

# Split into training and test sets, first 510750 for training data, rest for test
train_indices = collect(1:510750)
test_indices = collect(510751:514500)

function get_processed_data(args)
    labels = string.(X1.choice)
    features = Matrix(X1[:,2:end])'

    # Subract mean, divide by std dev for normed mean of 0 and std dev of 1.
    normed_features = normalise(features, dims=2)

    klasses = sort(unique(labels))
    onehot_labels = onehotbatch(labels, klasses)

    
    X_train = normed_features[:, train_indices]
    y_train = onehot_labels[:, train_indices]

    X_test = normed_features[:, test_indices]
    y_test = onehot_labels[:, test_indices]

    #repeat the data `args.repeat` times
    train_data = Iterators.repeated((X_train, y_train), args.repeat)
    test_data = (X_test,y_test)

    return train_data, test_data
end

# Accuracy Function
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))

# Function to build confusion matrix
function confusion_matrix(X, y, model)
    ŷ = onehotbatch(onecold(model(X)), 1:2)
    y * transpose(ŷ)
end

function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data = get_processed_data(args)

    # Declare model taking 27 features as inputs and outputting 2 probabiltiies
    model = Chain(
        Dense(27,27,relu),
        Dense(27, 27, relu),
        Dense(27, 20, relu),
        Dense(20, 10),
        softmax,
        Dense(10,2))
    #model = Chain(
    #    Dense(27, 27, relu),
    #    Dense(27, 2),
    #    softmax)

    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, Flux.params(model), train_data, optimiser)

    return model, test_data
end

function test(model, test)
    # Testing model performance on test data
    X_test, y_test = test
    accuracy_score = accuracy(X_test, y_test, model)

    println("\nAccuracy: $accuracy_score")

    # Sanity check.
    @assert accuracy_score > 0.1

    # To avoid confusion, here is the definition of a Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
    println("\nConfusion Matrix:\n")
    display(confusion_matrix(X_test, y_test, model))
    ##Loss function
    println("Loss test data")
    loss(x, y) = logitcrossentropy(model(x), y)
    display(loss(X_test,y_test))
end

cd(@__DIR__)
model, test_data = train()
test(model, test_data)



#Incorporating Covariates for Risk Preference
#LotShape already is measuring the risk of the lottery 

#Create a expected value variable for each lottery
raw_data.ExA = raw_data.Ha .* raw_data.pHa .+ raw_data.La .* (1 .- raw_data.pHa)   
raw_data.ExB = raw_data.Hb .* raw_data.pHb .+ raw_data.Lb .* (1 .- raw_data.pHb)
#and a variable equal to 1 if the expected value of B is higher than A
raw_data.ExB_best .= ifelse.(raw_data.ExB .> raw_data.ExA, 1, 0)   

#Create a expected value variable for each lottery
raw_data.ExHA = raw_data.Ha .* raw_data.pHa 
raw_data.ExHB = raw_data.Hb .* raw_data.pHb 
#and a variable equal to 1 if the expected value of B is higher than A
raw_data.ExHB_best .= ifelse.(raw_data.ExHB .> raw_data.ExHA, 1, 0)   
X1=raw_data

#we modify the train function to include more inputs in the dense part
function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data = get_processed_data(args)

    # Declare model taking 34 features as inputs and outputting 2 probabiltiies
    #model = Chain(Dense(33, 2))
    model = Chain(
        Dense(33,33,relu),
        Dense(33, 33, relu),
        Dense(33, 20, relu),
        Dense(20, 10),
        softmax,
        Dense(10,2))
    #model = Chain(
    #    Dense(27, 27, relu),
    #    Dense(27, 2),
    #    softmax)

    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, Flux.params(model), train_data, optimiser)

    return model, test_data
end

cd(@__DIR__)
model, test_data = train()
test(model, test_data)

#Incorporating Covariates for Risk Preference
# Create variables to measure attention of users
#This variables will be created in R
#these variables include:
#mean response time by game and by block. 
#a dummy equal to 1 if the response time is higher than the mean response time of the block

#Consistency with an option, we will estimate the probability of choosing an option by game and block
#We will also estimate the probability of choosing an option by game, block and subject 
#we create a dummy if the subject probability to choose an option is higher than the group probability
#We will also compare if the subject probability to choose an option is consistent with the probability of the group
#it is consitent if the difference between both probabilities is less than 25%
#See extra_var.R to see the creation of these new variables, the output of the code is attention_var.csv
#porsiacaso=raw_data 
extra = CSV.read("attention_var.csv", DataFrame)
extra = extra[:, 2:end] 
raw_data = hcat(raw_data, extra)
CSV.write("corr_data.csv", raw_data)


X1=raw_data

function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data = get_processed_data(args)

    # Declare model taking 40 features as inputs and outputting 2 probabiltiies
    #model = Chain(Dense(40, 2))
    model = Chain(
        Dense(40,40,relu),
        Dense(40, 40, relu),
        Dense(40, 20, relu),
        Dense(20, 10),
        softmax,
        Dense(10,2))
    #model = Chain(
    #    Dense(40, 40, relu),
    #    Dense(40, 2),
    #    softmax)

    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, Flux.params(model), train_data, optimiser)

    return model, test_data
end

cd(@__DIR__)
model, test_data = train()
test(model, test_data)


#FINAL MODEL 
#meanRT Moretime Forgone Condition block Trial Payoff Gender GameID Age RT
#Location B_more Order meanB meanB_subject Set Feedback consistent Ha

columns_to_exclude = [:meanRT, :Moretime, :Forgone, :Condition, :block, :Trial, :Payoff, :Gender, :GameID, :Age, :RT, :Location, :B_more, :Order, :meanB, :meanB_subject, :Set, :Feedback, :consistent, :Ha]
final_data = select(raw_data, Not(columns_to_exclude))

X1=final_data

function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data = get_processed_data(args)

    # Declare model taking 20 features as inputs and outputting 2 probabiltiies
    #model = Chain(Dense(20, 2))
    #model = Chain(
    #    Dense(20, 20, relu),
    #    Dense(20, 20, relu),
    #    Dense(20, 20, relu),
    #    Dense(20, 10),
    #    softmax,
    #    Dense(10,2))
    model = Chain(
        Dense(20, 20, relu),
        Dense(20, 10, relu),
        Dense(10, 2),
        softmax)

    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, Flux.params(model), train_data, optimiser)

    return model, test_data
end

cd(@__DIR__)
model, test_data = train()
test(model, test_data)
