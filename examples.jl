export train_mlp, train_param_deeponet
function train_mlp(; pltit=true,n_train = 1000, n_test = 200,batch_size = 100,learning_rate = 0.01, num_epochs = 5000, decay = 0, num_feat = 5,hidden = 50)  
    gr()
    Random.seed!(1)
    T = Float32
    # Define the function function 
    f(x) = sin(2.0 * π * x) + 0.5 * sin(16.0 * π * x)
    df(x) = 2.0 * π * cos(2.0 * π * x) + 8.0 * π * cos(16.0 * π * x)

    # Generate random input data for training and testing
    function generate_data(n_samples)
        x = rand(T, n_samples)
        y = T.(f.(x))
        return x, y
    end

    # Number of samples for training and testing
    f_feat = num_feat * 2 + 1
    B = reshape([(2^i) * π for i in 1:num_feat], :, 1)

    # Generate training data
    x_train_init, y_train = generate_data(n_train)

    # Generate testing data
    x_test_init, y_test = generate_data(n_test)

    # Reshape data for Flux model
    x_train_init = reshape(x_train_init, 1, n_train)
    x_train = vcat(x_train_init, sin.(B .* x_train_init), cos.(B .* x_train_init))
    y_train = reshape(y_train, 1, n_train)

    x_test_init = reshape(x_test_init, 1, n_test)
    x_test = vcat(x_test_init, sin.(B .* x_test_init), cos.(B .* x_test_init))
    y_test = reshape(y_test, 1, n_test)

    #Define and initialize the model
    model = Chain(
        Dense(f_feat => hidden, Flux.gelu),   # activation function inside layer
        Dense(hidden => hidden, Flux.gelu),
        Dense(hidden => 1)
    )

    # Define optimizers and learning rate schedulers, and training hyperparameters

    loss_function(outputs, targets) = Flux.Losses.mse(outputs, targets)
    optim = Flux.setup(Adam(learning_rate), model)

    sched = ParameterSchedulers.Stateful(
        ParameterSchedulers.CosAnneal(; l0 =learning_rate, l1 = 1e-5*learning_rate, period=div(num_epochs, 1))
    )
    total_params = sum(length,Flux.trainables(model))
    println("Total number of model parameters - $(total_params)")

    #Setup the training and testing dataloaders
    loader_train = Flux.DataLoader(
        (T.(x_train), Float32.(y_train));
        batchsize=batch_size,
        shuffle=true
    )
    loader_test =
        Flux.DataLoader((T.(x_test), Float32.(y_test)); batchsize=batch_size)

    # Training and testing loop
    for epoch in 1:num_epochs
        loss_train = 0
        loss_test = 0

        for (i, (x, y)) in enumerate(loader_train)
            loss, grads = Flux.withgradient(model) do m
                y_pred = m(x)
                return loss_function(y_pred, y)
            end
            loss_train = loss_train + loss
            Flux.update!(optim, model, grads[1])
        end

        nextlr = ParameterSchedulers.next!(sched) # advance schedule
        Flux.adjust!(optim, nextlr)

        for (i, (x, y)) in enumerate(loader_test)
            loss_t = loss_function(model(x), y)
            loss_test = loss_test + loss_t
        end

        if epoch % 20 == 0
            println(
                "epoch: $epoch, loss_train: $(loss_train/length(loader_train)),  loss_test: $(loss_test/length(loader_test)) "
            )
        end
    end

    if pltit
        # Uncomment if plots is installed
        idx = 1
        output_pred = model(T.(x_test))
        plt = plot()
        scatter!(plt,x_test[idx,:],y_test[idx,:],label = "Actual",color=:black)
        scatter!(plt,x_test[idx,:],output_pred[idx,:],label = "Pred",color=:red)
        display(plt)
    end
    return model, x_test, y_test
end


# Train a deeponet in a parameteric setting
function train_param_deeponet(; pltit=true,batch_size = 100,learning_rate = 1e-4,num_epochs = 5000,num_pts = 100,n_train = 20,n_test = 10,num_feat = 0,inp_bsize = 1,latent_size = 2,T = Float32)
    gr()
    inp_tsize = 2 * num_feat + 1
    hid_nt = 50
    hid_nb = 50
    trunk_neurons = (inp_tsize, hid_nt, hid_nt, latent_size)
    bnet_neurons = (inp_bsize, hid_nb, hid_nb, latent_size)
    
    Random.seed!(1)
    f(x, ξ) = (2.0 * sin(2.0 * π * x) + ξ * sin(16.0 * π * x)) / (2.0 + ξ)
    df(x, ξ) = (4.0 * π * cos(2.0 * π * x) + ξ * 16 * π * cos(16.0 * π * x)) / (2.0 + ξ) #gradient of the function
    

    # Generate random input data for training and testing
    function generate_data(n_samples, num_pts)
        x = LinRange(0, 1, num_pts)
        ξ = rand(1, n_samples)
        y = zeros(num_pts, n_samples)
        for i in 1:n_samples
            y[:, i] = T.(f.(x, ξ[1, i]))
        end
        return T.(reshape(x, 1, :)), T.(ξ), y
    end

    # Set the hyperparameters
    
    #B = reshape([(2^i)*π for i in 1:num_feat],:,1)

    # Generate training data
    x_train_init, ξ_train, y_train_init = generate_data(n_train, num_pts)

    # Generate testing data
    x_test_init, ξ_test, y_test_init = generate_data(n_test, num_pts)

    x_branch_train = reshape(repeat(ξ_train, num_pts, 1), 1, :)
    x_trunk_init = repeat(x_train_init, 1, n_train)
    x_trunk_train = x_trunk_init  #vcat(x_trunk_init,sin.(B.*x_trunk_init),cos.(B.*x_trunk_init))
    y_train = reshape(y_train_init, 1, :)

    x_branch_test = reshape(repeat(ξ_test, num_pts, 1), 1, :)
    x_testt_init = repeat(x_test_init, 1, n_test)
    x_trunk_test = x_testt_init  #vcat(x_testt_init,sin.(B.*x_testt_init),cos.(B.*x_testt_init))
    y_test = reshape(y_test_init, 1, :)

    loader_train = Flux.DataLoader(
        (T.(x_branch_train), Float32.(x_trunk_train), Float32.(y_train));
        batchsize=batch_size,
        shuffle=false
    )
    loader_test = Flux.DataLoader(
        (T.(x_branch_test), Float32.(x_trunk_test), Float32.(y_test));
        batchsize=batch_size
    )

    model = DeepONet(
        bnet_neurons,
        trunk_neurons;
        siren_branch=false,
        siren_dec=true, # Corrected from siren_dec
        act_branch=Flux.gelu,
        act_dec=sin_func # Corrected from act_dec
    )

    loss_function(outputs, targets) = Flux.Losses.mse(outputs, targets)
     optim = Flux.setup(Adam(learning_rate), model)

    sched = ParameterSchedulers.Stateful(
        ParameterSchedulers.CosAnneal(; l0 =learning_rate, l1 = 1e-6*learning_rate, period=div(num_epochs, 1))
    )
    total_params = sum(length,Flux.trainables(model))
    println("Total number of model parameters - $(total_params)")

    # Training and testing loop
    for epoch in 1:num_epochs
        loss_train = 0
        loss_test = 0
        for (i, (xb, xt, y)) in enumerate(loader_train)
            loss, grads = Flux.withgradient(model) do m
                y_pred = m(xb, xt)
                return loss_function(y_pred, y)
            end
            loss_train = loss_train + loss
            Flux.update!(optim, model, grads[1])
        end

        # NEW
        nextlr = ParameterSchedulers.next!(sched) # advance schedule
        Flux.adjust!(optim, nextlr)

        for (i, (xb, xt, y)) in enumerate(loader_test)
            loss_t = loss_function(model(xb, xt), y)
            loss_test = loss_test + loss_t
        end

        if epoch % 20 == 0
            println(
                "epoch: $epoch, loss_train: $(loss_train/length(loader_train)),  loss_test: $(loss_test/length(loader_test)) "
            )
        end
    end
    if pltit
        # Uncomment if plots is being used
        output_pred_test = reshape(model(x_branch_test,x_trunk_test),:,n_test)
        y_out_test = reshape(y_test,:,n_test)
        n_idx = 1
        plt = plot()
        plot!(plt,x_train_init[1,:],y_out_test[:,n_idx],label = "Actual",color=:black,linewidth=3)
        plot!(plt,x_train_init[1,:],output_pred_test[:,n_idx],linestyle =:dash,label="NN-Prediction", color=:red,linewidth=3)
        display(plt)
    end
    return model, x_branch_test, x_trunk_test, y_test
end