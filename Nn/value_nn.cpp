//
// Created by zhou on 19-6-10.
//


#include "value_nn.h"


//ValueNn::ValueNn(int street) : street(street) {
//    torch::autograd::GradMode::set_enabled(false);
//    switch (street) {
//        case 1: {
//            module = torch::jit::load(aux_net_file);
//            break;
//        }
//        case 2: {
//            module = torch::jit::load(flop_net_file);
//            break;
//        }
//        case 3: {
//            module = torch::jit::load(turn_net_file);
//            break;
//        }
//        default:
//            break;
//    }
//    module.to(device);
//};
//
//void ValueNn::get_value(torch::Tensor& inputs, torch::Tensor& outputs) {
//    torch::autograd::GradMode::set_enabled(false);
//    std::vector<torch::jit::IValue> _inputs;
//    _inputs.emplace_back(inputs);
//    outputs.copy_(module.forward(_inputs).toTensor());
//}

ValueNn::ValueNn(int street) : street(street) {
    torch::autograd::GradMode::set_enabled(false);
    middle_size = 500;
    switch (street) {
        case 1: {
            output_size = 169 * 2;
            input_size = output_size + 1;
            read_weights(aux_net_file);
            break;
        }
        case 2: {
            output_size = 1000 * 2;
            input_size = output_size + 1;
            read_weights(flop_net_file);
            break;
        }
        case 3: {
            output_size = 1000 * 2;
            input_size = output_size + 1;
            read_weights(turn_net_file);
            break;
        }
        default:
            break;
    }
};

void ValueNn::read_weights(const char *s) {
    torch::autograd::GradMode::set_enabled(false);

//    float w_0[input_size * middle_size];
//    float b_0[middle_size];
//    float alpha_0[middle_size];
//
//    float w_1[middle_size * middle_size];
//    float b_1[middle_size];
//    float alpha_1[middle_size];
//
//    float w_2[middle_size * middle_size];
//    float b_2[middle_size];
//    float alpha_2[middle_size];
//
//    float w_3[middle_size * middle_size];
//    float b_3[middle_size];
//    float alpha_3[middle_size];
//
//    float w_4[middle_size * output_size];
//    float b_4[output_size];
//
//    std::ifstream f_read(s, std::ios::binary);
//    f_read.read( reinterpret_cast<char *>(&w_0[0]),sizeof(w_0) );
//    f_read.read( reinterpret_cast<char *>(&b_0[0]),sizeof(b_0) );
//    f_read.read( reinterpret_cast<char *>(&alpha_0[0]),sizeof(alpha_0) );
//
//    f_read.read( reinterpret_cast<char *>(&w_1[0]),sizeof(w_1) );
//    f_read.read( reinterpret_cast<char *>(&b_1[0]),sizeof(b_1) );
//    f_read.read( reinterpret_cast<char *>(&alpha_1[0]),sizeof(alpha_1) );
//
//    f_read.read( reinterpret_cast<char *>(&w_2[0]),sizeof(w_2) );
//    f_read.read( reinterpret_cast<char *>(&b_2[0]),sizeof(b_2) );
//    f_read.read( reinterpret_cast<char *>(&alpha_2[0]),sizeof(alpha_2) );
//
//    f_read.read( reinterpret_cast<char *>(&w_3[0]),sizeof(w_3) );
//    f_read.read( reinterpret_cast<char *>(&b_3[0]),sizeof(b_3) );
//    f_read.read( reinterpret_cast<char *>(&alpha_3[0]),sizeof(alpha_3) );
//
//    f_read.read( reinterpret_cast<char *>(&w_4[0]),sizeof(w_4) );
//    f_read.read( reinterpret_cast<char *>(&b_4[0]),sizeof(b_4) );
//
//    f_read.close();

    auto w_0 = new float[input_size * middle_size];
    auto b_0 = new float[middle_size];
    auto alpha_0 = new float[middle_size];

    auto w_1 = new float[middle_size * middle_size];
    auto b_1 = new float[middle_size];
    auto alpha_1 = new float[middle_size];

    auto w_2 = new float[middle_size * middle_size];
    auto b_2 = new float[middle_size];
    auto alpha_2 = new float[middle_size];

    auto w_3 = new float[middle_size * middle_size];
    auto b_3 = new float[middle_size];
    auto alpha_3 = new float[middle_size];

    auto w_4 = new float[middle_size * output_size];
    auto b_4 = new float[output_size];

    std::ifstream f_read(s, std::ios::binary);
    f_read.read( reinterpret_cast<char *>(&w_0[0]), input_size * middle_size * sizeof(float) );
    f_read.read( reinterpret_cast<char *>(&b_0[0]), middle_size * sizeof(float) );
    f_read.read( reinterpret_cast<char *>(&alpha_0[0]), middle_size * sizeof(float) );

    f_read.read( reinterpret_cast<char *>(&w_1[0]), middle_size * middle_size * sizeof(float) );
    f_read.read( reinterpret_cast<char *>(&b_1[0]), middle_size * sizeof(float) );
    f_read.read( reinterpret_cast<char *>(&alpha_1[0]), middle_size * sizeof(float) );

    f_read.read( reinterpret_cast<char *>(&w_2[0]), middle_size * middle_size * sizeof(float) );
    f_read.read( reinterpret_cast<char *>(&b_2[0]), middle_size * sizeof(float) );
    f_read.read( reinterpret_cast<char *>(&alpha_2[0]), middle_size * sizeof(float) );

    f_read.read( reinterpret_cast<char *>(&w_3[0]), middle_size * middle_size * sizeof(float) );
    f_read.read( reinterpret_cast<char *>(&b_3[0]), middle_size * sizeof(float) );
    f_read.read( reinterpret_cast<char *>(&alpha_3[0]), middle_size * sizeof(float) );

    f_read.read( reinterpret_cast<char *>(&w_4[0]), middle_size * output_size * sizeof(float) );
    f_read.read( reinterpret_cast<char *>(&b_4[0]), output_size * sizeof(float) );

    f_read.close();

    w0 = torch::from_blob(w_0, {input_size, middle_size}, torch::kFloat32).to(device);
    b0 = torch::from_blob(b_0, {middle_size}, torch::kFloat32).to(device);
    alpha0 = torch::from_blob(alpha_0, {middle_size}, torch::kFloat32).to(device);

    w1 = torch::from_blob(w_1, {middle_size, middle_size}, torch::kFloat32).to(device);
    b1 = torch::from_blob(b_1, {middle_size}, torch::kFloat32).to(device);
    alpha1 = torch::from_blob(alpha_1, {middle_size}, torch::kFloat32).to(device);

    w2 = torch::from_blob(w_2, {middle_size, middle_size}, torch::kFloat32).to(device);
    b2 = torch::from_blob(b_2, {middle_size}, torch::kFloat32).to(device);
    alpha2 = torch::from_blob(alpha_2, {middle_size}, torch::kFloat32).to(device);

    w3 = torch::from_blob(w_3, {middle_size, middle_size}, torch::kFloat32).to(device);
    b3 = torch::from_blob(b_3, {middle_size}, torch::kFloat32).to(device);
    alpha3 = torch::from_blob(alpha_3, {middle_size}, torch::kFloat32).to(device);

    w4 = torch::from_blob(w_4, {middle_size, output_size}, torch::kFloat32).to(device);
    b4 = torch::from_blob(b_4, {output_size}, torch::kFloat32).to(device);

    delete[] w_0;
    delete[] b_0;
    delete[] alpha_0;

    delete[] w_1;
    delete[] b_1;
    delete[] alpha_1;

    delete[] w_2;
    delete[] b_2;
    delete[] alpha_2;

    delete[] w_3;
    delete[] b_3;
    delete[] alpha_3;

    delete[] w_4;
    delete[] b_4;
}

void ValueNn::prelu_block(torch::Tensor& x, torch::Tensor& alpha, torch::Tensor& out) {
    torch::autograd::GradMode::set_enabled(false);
    out.copy_(x.clamp(0) + alpha * (x - torch::abs(x)) * 0.5);
}

void ValueNn::get_value(torch::Tensor& inputs, torch::Tensor& outputs) {
    torch::autograd::GradMode::set_enabled(false);
    torch::Tensor tmp_slice = tmp.slice(0, 0, inputs.sizes()[0], 1);
    torch::Tensor m_slice = m.slice(0, 0, inputs.sizes()[0], 1);

    m_slice.copy_(torch::matmul(inputs, w0)+b0);
    prelu_block(m_slice, alpha0, tmp_slice);

    m_slice.copy_(torch::matmul(tmp_slice, w1) + b1);
    prelu_block(m_slice, alpha1, tmp_slice);

    m_slice.copy_(torch::matmul(tmp_slice, w2) + b2);
    prelu_block(m_slice, alpha2, tmp_slice);

    m_slice.copy_(torch::matmul(tmp_slice, w3) + b3);
    prelu_block(m_slice, alpha3, tmp_slice);

    outputs.copy_(torch::matmul(tmp_slice, w4) + b4);

    outputs.copy_(outputs - 0.5 * (outputs * inputs.slice(1, 0, output_size, 1)).sum(-1, true));
}

ValueNn& get_aux_nn() {
    static ValueNn aux_net(1);
    return aux_net;
}

ValueNn& get_flop_nn() {
    static ValueNn flop_net(2);
    return flop_net;
}

ValueNn& get_turn_nn() {
    static ValueNn turn_net(3);
    return turn_net;
}

