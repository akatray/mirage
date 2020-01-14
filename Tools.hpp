// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include "Config.hpp"
#include "Sample.hpp"
#include <fx/Types.hpp>
#include <fx/Image.hpp>
#include <fx/Files.hpp>
#include <stacks/stacks.hpp>
#include <wui.hpp>
#include <vector>

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Mirage: Tools
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace mir::tools
{
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Expand namespaces.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	using namespace fx;
	namespace stdfs = std::filesystem;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Update preview image box with image.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	auto updateImageBox ( wui::Control& _ImageBox, const str _Bitmap, const Image<u8>& _Image ) -> void
	{
		auto Img = Image<u8>();
	
		if(_Image.depth() == 3) Img = img::remap(_Image, {2, 1, 0}); // Remap from RGB to BGR for windows bitmap.
		else Img = img::fatten(_Image, 3); // If single channel: fatten to 3 channels.
	
		wui::updateBitmap(_Bitmap, Img.data()); // Update windows bitmap.
		_ImageBox.setBitmap(wui::getBitmap(_Bitmap)); // Update image box.
	}

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Convert raw data to image.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	auto makeImage ( const cfg::PRECISION* _Data, const bool _Color = false, const uMAX _Width = cfg::S_WIDTH, const uMAX _Height = cfg::S_HEIGHT )
	{
		auto ImgR32 = Image<cfg::PRECISION>();
		if(_Color) ImgR32 = Image<cfg::PRECISION>(_Width, _Height, 3); // Create color image.
		else ImgR32 = Image<cfg::PRECISION>(_Width, _Height, 1); // Create grayscale image.
	
		ImgR32.copyIn(_Data); // Copy data into image.

		return Image<u8>(ImgR32); // Convert to u8 and return.
	}

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Make color image from 3 samples.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	auto makeColorImage ( const std::vector<uMAX> _Idx, const std::vector<Sample<cfg::PRECISION, cfg::S_WIDTH, cfg::S_HEIGHT>>& _Samples )
	{
		auto Channels = std::vector<Image<cfg::PRECISION>>();
		

		Channels.emplace_back(Image<cfg::PRECISION>(cfg::S_WIDTH, cfg::S_HEIGHT, 1)); // Create empty image.
		Channels.back().copyIn(_Samples[_Idx[0]].Data); // Sample to image.

		Channels.emplace_back(Image<cfg::PRECISION>(cfg::S_WIDTH, cfg::S_HEIGHT, 1));
		Channels.back().copyIn(_Samples[_Idx[1]].Data);

		Channels.emplace_back(Image<cfg::PRECISION>(cfg::S_WIDTH, cfg::S_HEIGHT, 1));
		Channels.back().copyIn(_Samples[_Idx[2]].Data);


		return Image<u8>(img::merge(Channels)); // Merge images, convert to u8 and return.
	}

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Make color image from 3 samples that was processed trough stack.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	auto makeTransColorImage ( const std::vector<uMAX> _Idx, const std::vector<Sample<cfg::PRECISION, cfg::S_WIDTH, cfg::S_HEIGHT>>& _Samples, sx::Network<cfg::PRECISION>& _Net )
	{
		auto Channels = std::vector<Image<cfg::PRECISION>>();
		

		_Net.exe(_Samples[_Idx[0]].Data); // Execute network on sample.
		Channels.emplace_back(Image<cfg::PRECISION>(cfg::S_WIDTH, cfg::S_HEIGHT, 1)); // Create empty image.
		Channels.back().copyIn(_Net.back()->out()); // Copy networks output to image.

		_Net.exe(_Samples[_Idx[1]].Data);
		Channels.emplace_back(Image<cfg::PRECISION>(cfg::S_WIDTH, cfg::S_HEIGHT, 1));
		Channels.back().copyIn(_Net.back()->out());

		_Net.exe(_Samples[_Idx[2]].Data);
		Channels.emplace_back(Image<cfg::PRECISION>(cfg::S_WIDTH, cfg::S_HEIGHT, 1));
		Channels.back().copyIn(_Net.back()->out());


		return Image<u8>(img::merge(Channels)); // Merge images, convert to u8 and return.
	}

	// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Samples loader.
	// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	auto loadSamples ( const str _Name, std::vector<mir::Sample<cfg::PRECISION, cfg::S_WIDTH, cfg::S_HEIGHT>>& _Samples )
	{
		auto SamplesCache = std::ifstream(cfg::P_WORKSPACE + _Name + std::to_string(cfg::S_WIDTH) + "x"s + std::to_string(cfg::S_HEIGHT) + "_"s + nameof<cfg::PRECISION>() + ".cache"s, std::ios::binary); // Open cache file.

		if(SamplesCache.is_open()) // If open: load from it.
		{
			std::cout << "Loading samples [" << _Name << "] from cache... ";


			auto SamplesCount = uMAX(0);
			SamplesCache >> SamplesCount; // Load samples count.

			_Samples.reserve(SamplesCount); // Allocate memory.
		
			for(auto s = uMAX(0); s < SamplesCount; ++s)
			{
				_Samples.emplace_back();
				_Samples.back().load(SamplesCache);
			}
		

			std::cout << "Done!\n";
		}


		else // If failed to open create new one.
		{
			std::cout << "Cache for [" << _Name << "] is missing! Baking from images... ";
		

			auto Files = files::buildFileList(cfg::P_WORKSPACE + _Name + "/"s, true); // Collect files into list.
		
			for(auto& File : Files) // For each file.
			{
				try // Catch errors.
				{
					auto Img = Image<u8>(File.string()); // Load image.
					if((Img.width() != cfg::S_WIDTH) || (Img.height() != cfg::S_HEIGHT)) Img = img::resize(Img, cfg::S_WIDTH, cfg::S_HEIGHT); // Resize if image is not in processing size.
			
					auto ImgRaw = Image<cfg::PRECISION>(Img); // Convert image to format for training.
					auto ImgRawRawChannels = img::split(ImgRaw); // Splits channels into separate samples.

					for(auto s = uMAX(0); s < ImgRawRawChannels.size(); ++s)
					{
						_Samples.emplace_back();
						std::memcpy(_Samples.back().Data, ImgRawRawChannels[s].data(), ImgRawRawChannels[s].sizeInBytes());
					}
				}

				catch(const Error& e) // Skip file if there were error when processing.
				{
					std::cout << "Error while processing file: " << File << '\n';
					continue;
				}
			}
			

			auto NewCache = std::ofstream(cfg::P_WORKSPACE + _Name + std::to_string(cfg::S_WIDTH) + "x"s + std::to_string(cfg::S_HEIGHT) + "_"s + nameof<cfg::PRECISION>() + ".cache"s, std::ios::binary); // Open cache file.
			NewCache << uMAX(_Samples.size()); // Store samples count.
			for(auto& Sample : _Samples) Sample.store(NewCache); // Store samples.


			std::cout << "Baked [" << _Samples.size() << "] samples.\n";
		}
	}

	// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Used to collect all images from source, filter, resize and store on destination.
	// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	auto collectImages ( const str _Dst, const str _Src )
	{
		if(!stdfs::exists(cfg::P_WORKSPACE + _Dst)) stdfs::create_directory(cfg::P_WORKSPACE + _Dst); // Create destination if it does not exist.
		

		auto Files = files::buildFileList(cfg::P_WORKSPACE + _Src, true); // Build file list of all file in source.
		
		for(auto& File : Files) 	// Process each file.
		{
			try // Catch errors.
			{
				auto Img = Image<u8>(File.string()); // Load image.


				if(Img.depth() != 3) // Drop image if not of depth 3.
				{
					std::cout << "Did not had 3 channels: " << File << '\n';
					std::cout << "Target: 3 | Input: " << Img.depth() << "\n\n";
					continue; 
				}


				const auto Aspect = r64(Img.width()) / Img.height(); // Get image's aspect ratio.

				if(std::abs(Aspect - cfg::S_STORAGE_ASPECT) > 0.25) // Drop image if aspect ratio deviates too much.
				{
					std::cout << "Aspect ratio deviated too much: " << File << '\n';
					std::cout << "Target ratio: " << cfg::S_STORAGE_ASPECT << " | Input ratio: " << Aspect << " | Deviation: " << std::abs(Aspect - cfg::S_STORAGE_ASPECT) << "\n\n";
					continue; 
				}


				auto ImgThumb = img::resize(Img, 32, 32); // Resize to reduce pixel count to check.
				auto Variation = uMAX(0);
				
				for(auto p = uMAX(0); p < ImgThumb.size(); p +=3)
				{
					Variation += std::abs(i64(ImgThumb[p]) - ImgThumb[p+1]);
					Variation += std::abs(i64(ImgThumb[p]) - ImgThumb[p+2]);
					Variation += std::abs(i64(ImgThumb[p+1]) - ImgThumb[p+2]);
				}

				Variation /= ImgThumb.size();

				if(Variation < 5) // Drop if variation is to low.
				{
					std::cout << "Did not pass grayscale test: " << File << '\n';
					std::cout << "Target variation: ? < 5 | Input variation: " << Variation << "\n\n";
					continue;
				}


				while(true)
				{
					const auto FileName = cfg::P_WORKSPACE + _Dst + "/" + rng::getString(32) + ".jpg"s; // Generate name.
					if(stdfs::exists(FileName)) continue; // Reroll name if its already in use.

					Img = img::resize(Img, cfg::S_STORAGE_WIDTH, cfg::S_STORAGE_HEIGHT); // Resize image to storage size.
					Img.save(FileName, img::FileFormat::JPG); // Store to destination.
					
					break;
				}
				

			}

			catch(const Error& e) // Skip file if there were error when processing.
			{
				std::cout << "Error while processing file: " << File << '\n';
				continue;
			}
		}
	}
}
