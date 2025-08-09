-- fxmanifest.lua
fx_version 'cerulean'
game 'gta5'

author 'YourName'
description 'RSGCore whisky_bottle item client side'
version '1.0.0'

client_script 'client/main.lua'

-- config.lua
Config = {}

-- client/main.lua
local RSGCore = exports['rsg-core']:GetCoreObject()

RegisterNetEvent('rsg-core:client:onPlayerLoadItem', function(item)
    if item.name == 'whisky_bottle' then
        local playerPed = PlayerPedId()
        local animationDict = "mini@drinking@coffee"
        local animationName = "coffee_loop"

        RequestAnimDict(animationDict)
        while not HasAnimDictLoaded(animationDict) do
            Wait(100)
        end

        TaskPlayAnim(playerPed, animationDict, animationName, 8.0, -8.0, -1, 50, 0, false, false, false)

        local drinkSound = "coffee"
        PlaySoundFrontend(-1, drinkSound, "DLC_HEIST_PLANNING_BOARD_SOUNDS", true)

        RSGCore.Functions.Notify("Vous avez bu une bouteille de whisky", "success")

        TriggerServerEvent('rsg-core:server:useItem', 'whisky_bottle')
    end
end)

-- server/main.lua
RegisterServerEvent('rsg-core:server:useItem')
AddEventHandler('rsg-core:server:useItem', function(itemName)
    if itemName == 'whisky_bottle' then
        local src = source
        local player = RSGCore.Functions.GetPlayer(src)

        if player then
            player.Functions.RemoveItem('whisky_bottle', 1)
            player.Functions.AddItem('health_kit', 1)
            player.Functions.AddItem('stamina_kit', 1)
        end
    end
end)